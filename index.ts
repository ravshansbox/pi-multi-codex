import { loginOpenAICodex } from "@earendil-works/pi-ai/oauth";
import { type ExtensionAPI, type ExtensionCommandContext, type AuthStorage, type Theme, type OAuthCredential, type ApiKeyCredential, type AuthCredential } from "@earendil-works/pi-coding-agent";
import { matchesKey, type TUI, type KeybindingsManager, type Component } from "@earendil-works/pi-tui";

const ACCOUNT_PREFIX = "openai-codex-";
const ACTIVE_KEY = "openai-codex";
const FETCH_TIMEOUT_MS = 10_000;

const DEBUG = !!process.env.MULTI_CODEX_DEBUG;

function debug(...args: unknown[]): void {
	if (DEBUG) console.error("[multi-codex]", ...args);
}

interface CodexUsageWindow {
	pct: number;
	reset?: number;
}

interface CodexUsageResponse {
	rate_limit?: {
		primary_window?: {
			limit_window_seconds?: number;
			used_percent?: number;
			reset_at?: number;
		};
		secondary_window?: {
			limit_window_seconds?: number;
			used_percent?: number;
			reset_at?: number;
		};
	};
	plan_type?: string;
	email?: string;
}

async function fetchUsage(apiKey: string) {
	try {
		const res = await fetch("https://chatgpt.com/backend-api/wham/usage", {
			headers: {
				Authorization: `Bearer ${apiKey}`,
				"User-Agent": "pi-agent",
				Accept: "application/json",
			},
			signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
		});

		if (!res.ok) {
			debug("fetchUsage non-ok", res.status);
			return null;
		}

		const data = (await res.json()) as CodexUsageResponse;
		const windows: Record<string, { percent: number; reset?: number }> = {};
		const ONE_DAY_SECONDS = 86_400;

		for (const raw of [data.rate_limit?.primary_window, data.rate_limit?.secondary_window]) {
			if (!raw) continue;

			const durationSeconds: number = raw.limit_window_seconds ?? 0;
			const label =
				durationSeconds > 0 && durationSeconds < ONE_DAY_SECONDS
					? "primary"
					: "week";

			windows[label] = {
				percent: raw.used_percent || 0,
				reset: raw.reset_at ? raw.reset_at * 1000 : undefined,
			};
		}

		let plan: string | undefined;
		if (data.plan_type && data.plan_type !== "free") plan = data.plan_type;

		const email = typeof data.email === "string" ? data.email : undefined;

		return { windows, plan, email };
	} catch (err) {
		debug("fetchUsage error", err);
		return null;
	}
}

function parseEmailFromJwt(apiKey: string): string | undefined {
	try {
		const [, payloadB64] = apiKey.split(".");
		if (!payloadB64) return undefined;

		const decoded = JSON.parse(
			Buffer.from(payloadB64, "base64url").toString("utf8"),
		);

		return decoded?.["https://api.openai.com/profile"]?.email as string | undefined;
	} catch (err) {
		debug("parseEmailFromJwt error", err);
		return undefined;
	}
}

function makeAccountKey(index: number): string {
	return `${ACCOUNT_PREFIX}${index}`;
}

function getAccountKeys(authStorage: AuthStorage): string[] {
	return authStorage
		.list()
		.filter((key) => key.startsWith(ACCOUNT_PREFIX))
		.sort();
}

function getActiveAccountKey(authStorage: AuthStorage): string | undefined {
	const activeCredential = authStorage.get(ACTIVE_KEY);
	if (!activeCredential) return undefined;

	for (const accountKey of getAccountKeys(authStorage)) {
		const credential = authStorage.get(accountKey);
		if (
			credential &&
			credential.type === activeCredential.type &&
			(activeCredential.type === "oauth"
				? (credential as OAuthCredential).access === (activeCredential as OAuthCredential).access
				: (credential as ApiKeyCredential).key === (activeCredential as ApiKeyCredential).key)
		) {
			return accountKey;
		}
	}
	return undefined;
}

function isCredentialExpired(credential: any): boolean {
	const expires: unknown = credential?.expires;
	if (typeof expires !== "number") return true;
	return Date.now() >= expires;
}

async function getAccessToken(authStorage: AuthStorage, key: string): Promise<string | undefined> {
	const credential = authStorage.get(key);
	if (!credential) return undefined;

	if (credential.type === "api_key" && !isCredentialExpired(credential)) {
		return (credential as ApiKeyCredential).key;
	}

	return authStorage.getApiKey(key).catch((err: unknown) => {
		debug("getApiKey error", key, err);
		return undefined;
	});
}

type UsageColor = "error" | "warning" | "success";

interface UsageWindowRow {
	name: string;
	percent: number;
	reset?: number;
	color: UsageColor;
}

interface AccountRow {
	key: string;
	index: number;
	email: string;
	plan?: string;
	usageWindows: UsageWindowRow[];
	error?: string;
	active: boolean;
}

const USAGE_SORT_ORDER: Record<string, number> = { week: 0, primary: 1 };

class AccountList implements Component {
	private rows: AccountRow[] = [];
	private loading = true;
	private selectedIndex = 0;
	private tui: TUI;
	private theme: Theme;
	private done: (_: unknown) => void;
	private context: ExtensionCommandContext;
	private busy = "";

	constructor(
		tui: TUI,
		theme: Theme,
		_keybindings: KeybindingsManager,
		done: (_: unknown) => void,
		context: ExtensionCommandContext,
	) {
		this.tui = tui;
		this.theme = theme;
		this.done = done;
		this.context = context;
		void this.init();
	}

	private dim(text: string): string {
		return this.theme.fg("muted", text);
	}

	private bold(text: string): string {
		return this.theme.bold(text);
	}

	private accent(text: string): string {
		return this.theme.fg("accent", text);
	}

	private async init() {
		const authStorage = this.context.modelRegistry.authStorage;
		const accountKeys = getAccountKeys(authStorage);
		const activeKey = getActiveAccountKey(authStorage);
		const rows: AccountRow[] = [];

		for (const accountKey of accountKeys) {
			const accountIndex = parseInt(accountKey.slice(ACCOUNT_PREFIX.length), 10);
			const credential = authStorage.get(accountKey);
			if (!credential) {
				rows.push({
					key: accountKey,
					index: accountIndex,
					email: "unknown",
					usageWindows: [],
					error: "missing",
					active: accountKey === activeKey,
				});
				continue;
			}

			const apiKey = await getAccessToken(authStorage, accountKey);
			if (!apiKey) {
				rows.push({
					key: accountKey,
					index: accountIndex,
					email: "unknown",
					usageWindows: [],
					error: "auth expired",
					active: accountKey === activeKey,
				});
				continue;
			}

			const usage = await fetchUsage(apiKey);
			const email = parseEmailFromJwt(apiKey) ?? usage?.email;
			const usageWindows: UsageWindowRow[] = [];

			if (usage?.windows) {
				const entries = Object.entries(usage.windows).sort(
					(a, b) => (USAGE_SORT_ORDER[a[0]] ?? 99) - (USAGE_SORT_ORDER[b[0]] ?? 99),
				);
				for (const [windowName, windowData] of entries) {
					const remaining = 100 - windowData.percent;
					usageWindows.push({
						name: windowName,
						percent: windowData.percent,
						reset: windowData.reset,
						color: remaining <= 10 ? "error" : remaining <= 30 ? "warning" : "success",
					});
				}
			}

			rows.push({
				key: accountKey,
				index: accountIndex,
				email: email ?? "unknown",
				plan: usage?.plan,
				usageWindows,
				error: usage ? undefined : "fetch failed",
				active: accountKey === activeKey,
			});
		}

		this.rows = rows;

		const activeIndex = rows.findIndex((row) => row.active);
		if (activeIndex >= 0) this.selectedIndex = activeIndex;

		this.loading = false;
		this.tui.requestRender();
	}

	handleInput(event: string): void {
		if (this.busy) return;

		if (matchesKey(event, "escape")) {
			this.done(undefined);
			return;
		}

		if (matchesKey(event, "up") || event === "k") {
			this.selectedIndex = Math.max(0, this.selectedIndex - 1);
			this.tui.requestRender();
			return;
		}

		if (matchesKey(event, "down") || event === "j") {
			this.selectedIndex = Math.min(this.rows.length - 1, this.selectedIndex + 1);
			this.tui.requestRender();
			return;
		}

		if (matchesKey(event, "enter")) {
			void this.withBusy("switch", () => this.switchAccount());
			return;
		}

		if (event === "a") {
			void this.withBusy("add", () => this.addAccount());
			return;
		}

		if (matchesKey(event, "backspace") || matchesKey(event, "delete")) {
			void this.withBusy("remove", () => this.removeAccount());
		}
	}

	private async withBusy(label: string, action: () => Promise<void>) {
		this.busy = label;
		this.tui.requestRender();
		try {
			await action();
		} finally {
			this.busy = "";
			this.tui.requestRender();
		}
	}

	private async switchAccount() {
		const row = this.rows[this.selectedIndex];
		if (!row || row.active) return;

		const authStorage = this.context.modelRegistry.authStorage;
		const credential = authStorage.get(row.key);
		if (credential) {
			authStorage.set(ACTIVE_KEY, credential);
		}

		this.done(undefined);

		try {
			await this.context.reload();
		} catch (error) {
			this.context.ui.notify(
				`Reload failed: ${error instanceof Error ? error.message : String(error)}`,
				"error",
			);
		}
	}

	private async addAccount() {
		try {
			const credentials = await loginOpenAICodex({
				onAuth: ({ url, instructions }) => {
					this.context.ui.notify(`Open: ${url}`, "info");
					if (instructions) this.context.ui.notify(instructions, "info");

					void import("node:child_process").then(({ exec }) => {
						let openCmd: string;
						if (process.platform === "darwin") {
							openCmd = `open '${url}'`;
						} else if (process.platform === "win32") {
							openCmd = `start "" "${url}"`;
						} else {
							openCmd = `xdg-open '${url}'`;
						}
						exec(openCmd);
					});
				},
				onProgress: (msg: string) => this.context.ui.notify(msg, "info"),
				onPrompt: async ({ message }: { message: string }) => {
					const value = await this.context.ui.input(message);
					if (!value?.trim()) throw new Error("Cancelled");
					return value.trim();
				},
				originator: "pi",
			});

			const authStorage = this.context.modelRegistry.authStorage;

			let nextIndex = 0;
			for (const key of authStorage.list()) {
				if (key.startsWith(ACCOUNT_PREFIX)) {
					const existingIndex = parseInt(key.slice(ACCOUNT_PREFIX.length), 10);
					if (!isNaN(existingIndex) && existingIndex >= nextIndex) {
						nextIndex = existingIndex + 1;
					}
				}
			}

			const credential: AuthCredential = { ...credentials, type: "oauth" };
			authStorage.set(makeAccountKey(nextIndex), credential);
			authStorage.set(ACTIVE_KEY, credential);

			this.context.ui.notify(`Added & switched to [${nextIndex}]`, "info");
		} catch (error) {
			this.context.ui.notify(
				`Failed: ${error instanceof Error ? error.message : String(error)}`,
				"error",
			);
		}

		this.loading = true;
		void this.init().then(() => {
			this.tui.requestRender();
		});
	}

	private async removeAccount() {
		const row = this.rows[this.selectedIndex];
		if (!row) return;

		const authStorage = this.context.modelRegistry.authStorage;
		if (row.active) {
			authStorage.remove(ACTIVE_KEY);
		}
		authStorage.remove(row.key);

		this.loading = true;
		void this.init().then(() => {
			this.selectedIndex = Math.max(0, Math.min(this.selectedIndex, this.rows.length - 1));
			this.tui.requestRender();
		});
	}

	invalidate(): void {}
	dispose(): void {}

	render(width: number): string[] {
		const theme = this.theme;
		const innerWidth = width - 4;
		const horizontalLine = "─".repeat(width - 2);

		const boxLine = (content: string): string => {
			const visible = content.replace(/\x1b\[[0-9;]*m/g, "");
			const padding = " ".repeat(Math.max(0, innerWidth - visible.length));
			return this.dim("│ ") + content + padding + this.dim(" │");
		};

		const lines: string[] = [];

		if (this.busy) {
			lines.push(
				this.dim(`╭${horizontalLine}╮`),
				boxLine(this.bold(this.accent("multi-codex"))),
				this.dim(`├${horizontalLine}┤`),
				boxLine(`${this.busy}...`),
				this.dim(`╰${horizontalLine}╯`),
			);
			return lines;
		}

		lines.push(
			this.dim(`╭${horizontalLine}╮`),
			boxLine(this.bold(this.accent("multi-codex"))),
			this.dim(`├${horizontalLine}┤`),
		);

		if (this.loading) {
			lines.push(boxLine("loading..."));
		} else if (!this.rows.length) {
			lines.push(boxLine("no accounts"), boxLine(""), boxLine(this.dim("a  add account")));
		} else {
			for (let index = 0; index < this.rows.length; index++) {
				const row = this.rows[index];
				const isSelected = index === this.selectedIndex;

				const planLabel = row.plan ? theme.fg("accent", ` ${row.plan}`) : "";
				const cursor = isSelected ? theme.fg("accent", "▸ ") : "  ";
				const activeDot = row.active ? theme.fg("success", " ●") : "";
				lines.push(
					boxLine(
						`${cursor}${this.bold(`[${row.index}]`)} ${row.email}${planLabel}${activeDot}`,
					),
				);

				if (row.error) {
					lines.push(boxLine(this.dim(`   ${row.error}`)));
					continue;
				}

				for (const window of row.usageWindows) {
					const filled = Math.min(10, Math.round(window.percent / 10));
					const empty = 10 - filled;
					const bar = theme.fg(window.color, "█".repeat(filled)) + this.dim("░".repeat(empty));
					const resetLabel = window.reset
						? this.dim(` ${formatCountdown(new Date(window.reset))}`)
						: "";

					lines.push(
						boxLine(
							`   ${window.name.padEnd(7)} ${bar} ${window.percent.toFixed(0).padStart(3)}%${resetLabel}`,
						),
					);
				}
			}
		}

		lines.push(
			this.dim(`├${horizontalLine}┤`),
			boxLine(this.dim("↑↓ select  a add  ↵ switch  ⌫ remove  esc close")),
			this.dim(`╰${horizontalLine}╯`),
		);

		return lines;
	}
}

function formatCountdown(date: Date): string {
	const diffMs = date.getTime() - Date.now();
	if (diffMs < 0) return "now";

	const minutes = Math.floor(diffMs / 60_000);
	if (minutes < 60) return `in ${minutes}m`;

	const hours = Math.floor(minutes / 60);
	const remainingMinutes = minutes % 60;
	const minuteSuffix = remainingMinutes ? ` ${remainingMinutes}m` : "";

	if (hours < 24) return `in ${hours}h${minuteSuffix}`;

	return `in ${Math.floor(hours / 24)}d`;
}

export default function (pi: ExtensionAPI) {
	pi.on("session_start", async (_event, context) => {
		const authStorage = context.modelRegistry.authStorage;
		for (const key of authStorage.list()) {
			if (key.startsWith(ACCOUNT_PREFIX) || key === ACTIVE_KEY) {
				try {
					await getAccessToken(authStorage, key);
				} catch (err) {
					debug("session_start refresh failed", key, err);
				}
			}
		}
	});

	pi.on("message_end", async (event, context) => {
		if (event.message.role !== "assistant") return;

		const msg = event.message as any;
		if (msg.stopReason !== "error") return;

		const errorMessage: string = msg.errorMessage ?? "";
		if (!/usage.limit/i.test(errorMessage)) return;

		const authStorage = context.modelRegistry.authStorage;
		const accountKeys = getAccountKeys(authStorage);
		if (accountKeys.length <= 1) return;

		const activeKey = getActiveAccountKey(authStorage);

		const scored = (
			await Promise.all(
				accountKeys.map(async (key) => {
					const accessToken = await getAccessToken(authStorage, key);
					if (!accessToken) return null;

					const usage = await fetchUsage(accessToken);
					if (!usage?.windows) return null;

					const now = Date.now();
					const primary = usage.windows["primary"];
					const week = usage.windows["week"];

					const primaryPct =
						primary?.reset && primary.reset <= now ? 0 : (primary?.percent ?? 0);
					const weekPct =
						week?.reset && week.reset <= now ? 0 : (week?.percent ?? 0);

					const usagePct = Math.max(primaryPct, weekPct);
					const resetAt = Math.min(
						primary?.reset ?? Infinity,
						week?.reset ?? Infinity,
					);

					const credential = authStorage.get(key);
					if (!credential) return null;
					return { key, pct: usagePct, reset: resetAt, credential };
				}),
			)
		).filter(
			(
				s,
			): s is {
				key: string;
				pct: number;
				reset: number;
				credential: AuthCredential;
			} => s !== null,
		);

		if (scored.length === 0) return;

		scored.sort((a, b) => {
			const aExhausted = a.pct >= 100 ? 1 : 0;
			const bExhausted = b.pct >= 100 ? 1 : 0;
			if (aExhausted !== bExhausted) return aExhausted - bExhausted;
			return a.reset - b.reset;
		});

		const best = scored.find((s) => s.key !== activeKey) ?? scored[0];

		if (best.pct >= 100) {
			context.ui.notify("All accounts rate-limited — wait for reset", "error");
			return;
		}

		authStorage.set(ACTIVE_KEY, best.credential);

		const resetDate = new Date(best.reset);
		context.ui.notify(
			`Switched to [${best.key.slice(ACCOUNT_PREFIX.length)}] ` +
				`(resets ${resetDate.getHours()}:` +
				`${String(resetDate.getMinutes()).padStart(2, "0")})`,
			"warning",
		);

		const entries = context.sessionManager.getEntries();

		for (let i = entries.length - 1; i >= 0; i--) {
			const entry = entries[i];
			if (entry.type === "message" && entry.message.role === "user") {
				pi.sendUserMessage(entry.message.content, { deliverAs: "followUp" });
				break;
			}
		}
	});

	pi.registerCommand("multi-codex", {
		description: "Manage multiple OpenAI Codex accounts",
		handler: async (_args, context) => {
			await context.ui.custom((tui, theme, keybindings, done) =>
				new AccountList(tui, theme, keybindings, done, context),
			);
		},
	});
}

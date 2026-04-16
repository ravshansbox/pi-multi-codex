/**
 * Multi-Codex Provider Extension
 *
 * Manages multiple OpenAI Codex accounts with smart routing based on
 * usage limits.
 *
 * Commands:
 *   /multi-codex add          Add OpenAI Codex account
 *   /multi-codex list         List accounts with usage
 *   /multi-codex remove <index>  Remove account
 *   /multi-codex usage        Show usage stats
 */

import {
	type Api,
	type AssistantMessageEventStream,
	type Context,
	createAssistantMessageEventStream,
	type Model,
	type SimpleStreamOptions,
	streamSimple,
} from "@mariozechner/pi-ai";
import { loginOpenAICodex, refreshOpenAICodexToken } from "@mariozechner/pi-ai/oauth";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { createHash } from "node:crypto";
import { exec } from "node:child_process";
import { matchesKey } from "@mariozechner/pi-tui";

// =============================================================================
// Rate Limit Classification
// =============================================================================

type RateLimitReason =
	| "QUOTA_EXHAUSTED"
	| "RATE_LIMIT_EXCEEDED"
	| "MODEL_CAPACITY_EXHAUSTED"
	| "SERVER_ERROR"
	| "UNKNOWN";

function parseRateLimitReason(errorMessage: string): RateLimitReason {
	const lower = errorMessage.toLowerCase();

	if (
		lower.includes("capacity") ||
		lower.includes("overloaded") ||
		lower.includes("529") ||
		lower.includes("503") ||
		lower.includes("resource exhausted")
	) {
		return "MODEL_CAPACITY_EXHAUSTED";
	}

	if (
		lower.includes("per minute") ||
		lower.includes("rate limit") ||
		lower.includes("too many requests")
	) {
		return "RATE_LIMIT_EXCEEDED";
	}

	if (lower.includes("exhausted") || lower.includes("quota") || lower.includes("usage limit")) {
		return "QUOTA_EXHAUSTED";
	}

	if (lower.includes("500") || lower.includes("internal error")) {
		return "SERVER_ERROR";
	}

	return "UNKNOWN";
}

function calculateRateLimitBackoffMs(reason: RateLimitReason): number {
	switch (reason) {
		case "QUOTA_EXHAUSTED":
			return 30 * 60 * 1000; // 30 min
		case "RATE_LIMIT_EXCEEDED":
			return 30 * 1000; // 30s
		case "MODEL_CAPACITY_EXHAUSTED":
			return 45 * 1000 + Math.random() * 30 * 1000; // 45-75s with jitter
		case "SERVER_ERROR":
			return 20 * 1000; // 20s
		default:
			return 30 * 60 * 1000; // conservative default: 30 min
	}
}

// =============================================================================
// Constants
// =============================================================================

const STORE_PATH = path.join(os.homedir(), ".pi", "agent", "multi-codex-auth.json");
const EXPIRY_SKEW_MS = 60_000;
const USAGE_STALE_MS = 5 * 60 * 1000; // 5 minutes
const FETCH_TIMEOUT_MS = 10000;

const PROVIDER_CONFIG = {
	provider: "openai-codex-multi",
	api: "openai-codex-multi-api" as Api,
	baseUrl: "https://chatgpt.com/backend-api",
	models: [
		{
			id: "gpt-5.4",
			name: "GPT-5.4",
			reasoning: true,
			input: ["text", "image"] as const,
			cost: { input: 2.5, output: 15, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 1050000,
			maxTokens: 128000,
		},
	],
} as const;

const STORE_KEY = PROVIDER_CONFIG.provider;

// =============================================================================
// Types
// =============================================================================

interface StoredAccount {
	access: string;
	refresh?: string;
	expires?: number;
	email?: string;
	addedAt: number;
	lastRefreshedAt?: number;
	usage?: {
		fetchedAt: number;
		windows: Record<string, { usedPercent: number; resetAt?: number }>;
	};
	depleted?: {
		until: number;
		reason: RateLimitReason;
		window: string;
	};
}

interface UsageSnapshot {
	index: number;
	email?: string;
	plan?: string;
	windows: Record<string, { usedPercent: number; resetAt?: number }>;
	error?: string;
}

interface UIContext {
	ui: any;
	cwd: string;
	sessionManager: any;
}

type AccountStore = { accounts: StoredAccount[] };

// =============================================================================
// Store Management
// =============================================================================

function ensureDir(p: string): void {
	fs.mkdirSync(path.dirname(p), { recursive: true });
}

function loadStore(): AccountStore {
	try {
		if (fs.existsSync(STORE_PATH)) {
			const raw = JSON.parse(fs.readFileSync(STORE_PATH, "utf-8")) as any;
			const store: AccountStore = raw.accounts ? raw : { accounts: raw[STORE_KEY]?.accounts ?? [] };
			// Clean up expired depleted entries
			const now = Date.now();
			for (const acc of store.accounts) {
				if (acc.depleted && acc.depleted.until <= now) {
					delete acc.depleted;
				}
			}
			return store;
		}
	} catch {
		// ignore
	}
	return { accounts: [] };
}

function saveStore(store: AccountStore): void {
	ensureDir(STORE_PATH);
	fs.writeFileSync(STORE_PATH, JSON.stringify(store, null, 2));
}

function markDepleted(index: number, entry: { until: number; reason: RateLimitReason; window: string }): void {
	const store = loadStore();
	if (index >= 0 && index < store.accounts.length) {
		store.accounts[index].depleted = entry;
		saveStore(store);
	}
}

// =============================================================================
// Session Key
// =============================================================================

let currentSessionKey = "no-session";

function getSessionKey(ctx: UIContext): string {
	return ctx.sessionManager?.getSessionFile?.() ?? ctx.sessionManager?.getLeafId?.() ?? `ephemeral:${ctx.cwd}`;
}

function hashIndex(sessionKey: string, accountCount: number): number {
	if (accountCount <= 0) return 0;
	const digest = createHash("sha256").update(sessionKey).digest();
	return digest.readUInt32BE(0) % accountCount;
}

function getAccountOrder(sessionKey: string, accountCount: number): number[] {
	if (accountCount <= 0) return [];
	const start = hashIndex(sessionKey, accountCount);
	return Array.from({ length: accountCount }, (_, i) => (start + i) % accountCount);
}

// =============================================================================
// Token Management
// =============================================================================

async function refreshToken(index: number, account: StoredAccount): Promise<StoredAccount | null> {
	if (!account.refresh) return account.access ? account : null;

	try {
		const refreshed = await refreshOpenAICodexToken(account.refresh);

		const updated: StoredAccount = {
			...account,
			access: refreshed.access,
			refresh: refreshed.refresh || account.refresh,
			expires: refreshed.expires,
			lastRefreshedAt: Date.now(),
		};

		const store = loadStore();
		store.accounts[index] = updated;
		saveStore(store);

		return updated;
	} catch {
		return account.access ? account : null;
	}
}

async function refreshAccountIfNeeded(index: number): Promise<StoredAccount | null> {
	const store = loadStore();
	const account = store.accounts[index];
	if (!account) return null;

	const now = Date.now();
	const needsRefresh = !account.expires || account.expires - EXPIRY_SKEW_MS <= now;

	if (!needsRefresh) return account;
	return refreshToken(index, account);
}

// =============================================================================
// Usage Fetching
// =============================================================================

async function fetchWithTimeout(url: string, init?: RequestInit): Promise<Response> {
	const controller = new AbortController();
	const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
	try {
		return await fetch(url, { ...init, signal: controller.signal });
	} finally {
		clearTimeout(timer);
	}
}

async function fetchUsage(account: StoredAccount): Promise<{ windows: Record<string, { usedPercent: number; resetAt?: number }>; email?: string; plan?: string } | null> {
	try {
		const headers: Record<string, string> = {
			Authorization: `Bearer ${account.access}`,
			"User-Agent": "pi-agent",
			Accept: "application/json",
		};
		if ((account as any).accountId) {
			headers["ChatGPT-Account-Id"] = (account as any).accountId;
		}

		const res = await fetchWithTimeout("https://chatgpt.com/backend-api/wham/usage", {
			headers,
		});

		if (!res.ok) return null;

		const data = (await res.json()) as any;
		const windows: Record<string, { usedPercent: number; resetAt?: number }> = {};

		if (data.rate_limit?.primary_window) {
			const pw = data.rate_limit.primary_window;
			windows["primary"] = {
				usedPercent: pw.used_percent || 0,
				resetAt: pw.reset_at ? new Date(pw.reset_at * 1000).getTime() : undefined,
			};
		}

		if (data.rate_limit?.secondary_window) {
			const sw = data.rate_limit.secondary_window;
			windows["week"] = {
				usedPercent: sw.used_percent || 0,
				resetAt: sw.reset_at ? new Date(sw.reset_at * 1000).getTime() : undefined,
			};
		}

		let plan = data.plan_type;
		if (data.credits?.balance !== undefined && data.credits.balance !== null) {
			const balance = typeof data.credits.balance === "number" ? data.credits.balance : parseFloat(data.credits.balance) || 0;
			if (balance > 0) plan = plan ? `${plan} ($${balance.toFixed(2)})` : `$${balance.toFixed(2)}`;
		}

		// Fetch email
		let email: string | undefined;
		try {
			const meRes = await fetchWithTimeout("https://api.openai.com/v1/me", {
				headers: { Authorization: `Bearer ${account.access}` },
			});
			if (meRes.ok) {
				const me = (await meRes.json()) as any;
				email = me?.email;
			}
		} catch {
			// ignore
		}

		return { windows, email, plan };
	} catch {
		return null;
	}
}

async function refreshUsage(index: number, account: StoredAccount): Promise<void> {
	const usage = await fetchUsage(account);
	if (!usage) return;

	const updated: StoredAccount = {
		...account,
		usage: {
			fetchedAt: Date.now(),
			windows: usage.windows,
		},
		email: usage.email || account.email,
	};

	const store = loadStore();
	store.accounts[index] = updated;
	saveStore(store);
}

// =============================================================================
// Smart Account Selection
// =============================================================================

async function selectAccount(
	sessionKey: string,
): Promise<{ account: StoredAccount; index: number } | null> {
	const store = loadStore();
	const accounts = store.accounts;
	const now = Date.now();

	const candidates: { account: StoredAccount; index: number; weekPercent: number }[] = [];

	for (let i = 0; i < accounts.length; i++) {
		const acc = accounts[i];
		if (!acc?.access) continue;

		if (acc.expires && acc.expires - EXPIRY_SKEW_MS <= now) continue;
		if (acc.depleted && acc.depleted.until > now) continue;

		const weekPercent = acc.usage?.windows["week"]?.usedPercent ?? 0;
		candidates.push({ account: acc, index: i, weekPercent });
	}

	if (candidates.length === 0) return null;

	candidates.sort((a, b) => a.weekPercent - b.weekPercent);

	const topPercent = candidates[0].weekPercent;
	const top = candidates.filter((c) => c.weekPercent === topPercent);

	let selected: { account: StoredAccount; index: number };
	if (top.length > 1) {
		const hashIdx = hashIndex(sessionKey, top.length);
		selected = top[hashIdx];
	} else {
		selected = candidates[0];
	}

	let account = await refreshAccountIfNeeded(selected.index);
	if (!account) return null;

	const usageAge = Date.now() - (account.usage?.fetchedAt ?? 0);
	if (usageAge >= USAGE_STALE_MS) {
		await refreshUsage(selected.index, account);
		const updatedStore = loadStore();
		account = updatedStore.accounts[selected.index] ?? account;
	}

	return { account, index: selected.index };
}

// =============================================================================
// OAuth Helpers
// =============================================================================

async function promptInput(ctx: UIContext, message: string): Promise<string> {
	const value = await ctx.ui.input(message);
	if (typeof value !== "string" || !value.trim()) throw new Error("Input cancelled");
	return value.trim();
}

async function openBrowser(url: string): Promise<void> {
	const platform = process.platform;
	const command =
		platform === "darwin"
			? `open '${url.replace(/'/g, `'\\''`)}'`
			: platform === "win32"
				? `start "" "${url.replace(/"/g, '""')}"`
				: `xdg-open '${url.replace(/'/g, `'\\''`)}'`;

	await new Promise<void>((resolve) => {
		exec(command, () => resolve());
	});
}

async function addAccount(ctx: UIContext): Promise<StoredAccount> {
	const credentials = await loginOpenAICodex({
		onAuth: ({ url, instructions }) => {
			void openBrowser(url);
			ctx.ui.notify("Open this URL in your browser:", "info");
			ctx.ui.notify(url, "info");
			if (instructions) ctx.ui.notify(instructions, "info");
		},
		onPrompt: ({ message }) => promptInput(ctx, message),
		onProgress: (message) => ctx.ui.notify(message, "info"),
		originator: "pi",
	});

	let email: string | undefined;
	try {
		const res = await fetch("https://api.openai.com/v1/me", {
			headers: { Authorization: `Bearer ${credentials.access}` },
		});
		if (res.ok) {
			const data = (await res.json()) as any;
			email = data?.email;
		}
	} catch {
		// ignore
	}

	return {
		access: credentials.access,
		refresh: credentials.refresh,
		expires: credentials.expires,
		email,
		addedAt: Date.now(),
	};
}

// =============================================================================
// Command Handler
// =============================================================================

async function handleCommand(args: string, ctx: UIContext): Promise<void> {
	const parts = (args || "").trim().split(/\s+/).filter(Boolean);
	const subcommand = parts[0]?.toLowerCase();
	const subArgs = parts.slice(1);
	const sessionKey = getSessionKey(ctx);

	switch (subcommand) {
		case "add": {
			let account: StoredAccount;
			try {
				account = await addAccount(ctx);
			} catch (e) {
				ctx.ui.notify(`Failed to add account: ${e}`, "error");
				return;
			}

			const store = loadStore();
			store.accounts.push(account);
			saveStore(store);

			const idx = store.accounts.length - 1;
			ctx.ui.notify(
				`OpenAI Codex account added at index ${idx}${account.email ? ` (${account.email})` : ""}`,
				"success",
			);

			refreshUsage(idx, account).catch(() => {});
			return;
		}

		case "list": {
			const store = loadStore();
			const now = Date.now();
			const lines: string[] = [];
			let hasAny = false;

			for (let i = 0; i < store.accounts.length; i++) {
				hasAny = true;
				const acc = store.accounts[i];
				const email = acc.email ?? "unknown";
				const weekPercent = acc.usage?.windows["week"]?.usedPercent;

				let status = "";
				if (acc.depleted && acc.depleted.until > now) {
					const remaining = Math.max(0, Math.ceil((acc.depleted.until - now) / 60000));
					status = ` [depleted ${remaining}m]`;
				} else if (acc.expires && acc.expires < now) {
					status = " [expired]";
				} else if (weekPercent !== undefined) {
					status = ` ${weekPercent >= 100 ? "🔴" : weekPercent >= 80 ? "🟡" : "🟢"} ${weekPercent.toFixed(0)}%`;
				}

				lines.push(`[${i}] ${email}${status}`);
			}

			if (!hasAny) {
				lines.push("(no accounts)");
			}

			ctx.ui.notify(lines.join("\n"), "info");
			return;
		}

		case "remove": {
			const rawIndex = subArgs[0];
			const index = Number(rawIndex);

			if (!Number.isInteger(index)) {
				ctx.ui.notify("Usage: /multi-codex remove <index>", "error");
				return;
			}

			const store = loadStore();
			if (index < 0 || index >= store.accounts.length) {
				ctx.ui.notify(`No account at index ${index}`, "error");
				return;
			}

			const removed = store.accounts[index];
			const ok = await ctx.ui.confirm(
				"Remove OpenAI Codex account",
				`Delete [${index}] ${removed.email ?? "unknown"}?`,
			);
			if (!ok) return;

			store.accounts.splice(index, 1);
			saveStore(store);

			ctx.ui.notify(`Removed OpenAI Codex account [${index}]${removed.email ? ` (${removed.email})` : ""}`, "success");
			return;
		}

		case "usage": {
			await ctx.ui.custom((tui: any, theme: any, _kb: any, done: () => void) => {
				return new UsageComponent(tui, theme, () => done(), ctx);
			});
			return;
		}

		case "":
		case undefined: {
			ctx.ui.notify(
				`Usage: /multi-codex [add|list|remove|usage]\n` +
				`  add              Add OpenAI Codex account\n` +
				`  list             List accounts with usage\n` +
				`  remove <index>   Remove account\n` +
				`  usage            Show usage stats\n` +
				`\nUsage is refreshed automatically.`,
				"info",
			);
			return;
		}

		default: {
			ctx.ui.notify(`Unknown command: ${subcommand}`, "error");
			return;
		}
	}
}

function getArgumentCompletions(prefix: string): { value: string; label: string }[] | null {
	const trimmed = prefix.trimStart();
	const parts = trimmed.split(/\s+/).filter(Boolean);
	const endsWithSpace = prefix.endsWith(" ");
	const store = loadStore();

	if (parts.length === 0) {
		return [
			{ value: "add", label: "add" },
			{ value: "list", label: "list" },
			{ value: "remove", label: "remove" },
			{ value: "usage", label: "usage" },
		];
	}

	if (parts.length === 1 && !endsWithSpace) {
		const subcommands = ["add", "list", "remove", "usage"];
		return subcommands.filter((s) => s.startsWith(parts[0])).map((s) => ({ value: s, label: s }));
	}

	if (parts[0] === "remove" && parts.length === 2 && !endsWithSpace) {
		return store.accounts
			.map((_, i) => String(i))
			.filter((i) => i.startsWith(parts[1]))
			.map((i) => ({ value: `remove ${i}`, label: i }));
	}

	return null;
}

// =============================================================================
// Usage UI Component
// =============================================================================

class UsageComponent {
	private usages: UsageSnapshot[] = [];
	private loading = true;
	private tui: { requestRender: () => void };
	private theme: any;
	private onClose: () => void;
	private ctx: UIContext;

	constructor(tui: { requestRender: () => void }, theme: any, onClose: () => void, ctx: UIContext) {
		this.tui = tui;
		this.theme = theme;
		this.onClose = onClose;
		this.ctx = ctx;
		this.load();
	}

	private async load() {
		const store = loadStore();
		const snapshots: UsageSnapshot[] = [];

		for (let i = 0; i < store.accounts.length; i++) {
			const acc = store.accounts[i];
			if (!acc?.access) continue;

			const refreshed = await refreshAccountIfNeeded(i);
			if (!refreshed) continue;

			const usageData = await fetchUsage(refreshed);

			snapshots.push({
				index: i,
				email: acc.email,
				plan: usageData?.plan,
				windows: usageData?.windows ?? acc.usage?.windows ?? {},
				error: usageData === null ? "fetch failed" : undefined,
			});
		}

		this.usages = snapshots;
		this.loading = false;
		this.tui.requestRender();
	}

	handleInput(data: string): void {
		if (matchesKey(data, "escape")) {
			this.onClose();
		}
	}

	invalidate(): void {}

	render(width: number): string[] {
		const t = this.theme;
		const dim = (s: string) => t.fg("muted", s);
		const bold = (s: string) => t.bold(s);
		const accent = (s: string) => t.fg("accent", s);

		const totalW = width;
		const innerW = totalW - 4;
		const hLine = "─".repeat(totalW - 2);

		const box = (content: string) => {
			const contentW = this.visibleWidth(content);
			const pad = Math.max(0, innerW - contentW);
			return dim("│ ") + content + " ".repeat(pad) + dim(" │");
		};

		const lines: string[] = [];
		lines.push(dim(`╭${hLine}╮`));
		lines.push(box(bold(accent("multi-codex usage"))));
		lines.push(dim(`├${hLine}┤`));

		if (this.loading) {
			lines.push(box("loading..."));
		} else if (this.usages.length === 0) {
			lines.push(box("no accounts configured"));
			lines.push(box(""));
			lines.push(box(dim("/multi-codex add")));
		} else {
			for (const u of this.usages) {
				const email = u.email ?? "unknown";
				const plan = u.plan ? dim(` (${u.plan})`) : "";
				lines.push(box(`${bold(`[${u.index}]`)} ${email}${plan}`));

				if (u.error) {
					lines.push(box(dim(`  ${u.error}`)));
					continue;
				}

				const sortedWindows = Object.entries(u.windows).sort((a, b) => {
					const order = { week: 0, primary: 1 };
					const oa = order[a[0] as keyof typeof order] ?? 99;
					const ob = order[b[0] as keyof typeof order] ?? 99;
					return oa - ob;
				});

				for (const [windowName, data] of sortedWindows) {
					const used = Math.max(0, Math.min(100, data.usedPercent));
					const remaining = Math.max(0, 100 - used);
					const barW = 10;
					const filled = Math.min(barW, Math.round((used / 100) * barW));
					const empty = barW - filled;

					const color = remaining <= 10 ? "error" : remaining <= 30 ? "warning" : "success";
					const bar = t.fg(color, "█".repeat(filled)) + dim("░".repeat(empty));

					const reset = data.resetAt ? dim(` ${this.formatReset(new Date(data.resetAt))}`) : "";
					lines.push(box(`  ${windowName.padEnd(7)} ${bar} ${used.toFixed(0).padStart(3)}%${reset}`));
				}

				lines.push(box(""));
			}
		}

		lines.push(dim(`├${hLine}┤`));
		lines.push(box(dim("press esc to close")));
		lines.push(dim(`╰${hLine}╯`));

		return lines;
	}

	private visibleWidth(s: string): number {
		return s.replace(/\x1b\[[0-9;]*m/g, "").length;
	}

	private formatReset(date: Date): string {
		const diffMs = date.getTime() - Date.now();
		if (diffMs < 0) return "now";
		const diffMins = Math.floor(diffMs / 60000);
		if (diffMins < 60) return `${diffMins}m`;
		const hours = Math.floor(diffMins / 60);
		const mins = diffMins % 60;
		if (hours < 24) return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
		const days = Math.floor(hours / 24);
		return `${days}d`;
	}

	dispose(): void {}
}

// =============================================================================
// Streaming with Smart Routing
// =============================================================================

function streamMultiProvider(
	model: Model<Api>,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream {
	const stream = createAssistantMessageEventStream();

	(async () => {
		const store = loadStore();
		const accounts = store.accounts;
		const order = getAccountOrder(currentSessionKey, accounts.length);
		const now = Date.now();
		let lastError: any = null;

		for (const index of order) {
			const acc = accounts[index];
			if (!acc?.access) continue;

			if (acc.expires && acc.expires - EXPIRY_SKEW_MS <= now) continue;
			if (acc.depleted && acc.depleted.until > now) continue;

			const account = await refreshAccountIfNeeded(index);
			if (!account) continue;

			const apiKey = account.access;
			const inner = streamSimple(
				{
					...model,
					api: "openai-codex-responses" as Api,
					provider: PROVIDER_CONFIG.provider,
				} as any,
				context,
				{ ...options, apiKey },
			);

			const buffered: any[] = [];
			let committed = false;

			for await (const rawEvent of inner as any) {
				const event = remapEvent(rawEvent, model);
				if (!committed) {
					if (event.type === "start") {
						buffered.push(event);
						continue;
					}
					if (event.type === "error") {
						const errorMsg = event.error?.errorMessage ?? "";
						const reason = parseRateLimitReason(errorMsg);
						if (reason !== "UNKNOWN" || /429|rate.?limit/i.test(errorMsg)) {
							const windowKey =
								reason === "QUOTA_EXHAUSTED"
									? "week"
									: "primary";

							let until = account.usage?.windows[windowKey]?.resetAt;
							if (!until || until <= Date.now()) {
								until = Date.now() + calculateRateLimitBackoffMs(reason);
							}

							markDepleted(index, {
								window: windowKey,
								until,
								reason,
							});
						}
						lastError = event;
						break;
					}
					committed = true;
					for (const pending of buffered) stream.push(pending);
				}
				stream.push(event);
			}

			if (committed) {
				stream.end();
				return;
			}
		}

		if (lastError) {
			stream.push(lastError);
			stream.end();
			return;
		}

		stream.push({
			type: "error",
			reason: "error",
			error: {
				role: "assistant",
				content: [],
				api: model.api,
				provider: model.provider,
				model: model.id,
				usage: {
					input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				stopReason: "error",
				errorMessage: "No usable OpenAI Codex accounts",
				timestamp: Date.now(),
			},
		});
		stream.end();
	})();

	return stream;
}

function remapEvent(event: any, model: Model<Api>) {
	if (!event || typeof event !== "object") return event;
	const remapped = { ...event };
	if ("partial" in remapped && remapped.partial?.role === "assistant") {
		remapped.partial = { ...remapped.partial, api: model.api, provider: model.provider, model: model.id };
	}
	if ("message" in remapped && remapped.message?.role === "assistant") {
		remapped.message = { ...remapped.message, api: model.api, provider: model.provider, model: model.id };
	}
	if ("error" in remapped && remapped.error?.role === "assistant") {
		remapped.error = { ...remapped.error, api: model.api, provider: model.provider, model: model.id };
	}
	return remapped;
}

// =============================================================================
// Extension Entry Point
// =============================================================================

export default function (pi: ExtensionAPI) {
	pi.on("session_start", async (_event, ctx) => {
		currentSessionKey = getSessionKey(ctx as UIContext);
	});

	pi.on("before_agent_start", async (_event, ctx) => {
		currentSessionKey = getSessionKey(ctx as UIContext);
	});

	pi.on("before_provider_request", async (_event, ctx) => {
		currentSessionKey = getSessionKey(ctx as UIContext);
	});

	pi.registerProvider(PROVIDER_CONFIG.provider, {
		baseUrl: PROVIDER_CONFIG.baseUrl,
		apiKey: "__multi_account_internal__",
		api: PROVIDER_CONFIG.api,
		models: PROVIDER_CONFIG.models,
		streamSimple: streamMultiProvider as any,
	});

	pi.registerCommand("multi-codex", {
		description: "Manage multi-account OpenAI Codex providers",
		getArgumentCompletions,
		handler: async (args, ctx) => handleCommand(args, ctx as UIContext),
	});
}

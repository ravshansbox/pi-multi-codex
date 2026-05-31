import { loginOpenAICodex } from "@earendil-works/pi-ai/oauth";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { matchesKey } from "@earendil-works/pi-tui";

const ACCOUNT_PREFIX = "openai-codex-";
const ACTIVE_KEY = "openai-codex";
const ACTIVE_POINTER_KEY = "openai-codex-active";
const FETCH_TIMEOUT_MS = 10_000;

const DEBUG = !!process.env.MULTI_CODEX_DEBUG;

function debug(...args: unknown[]): void {
  if (DEBUG) console.error("[multi-codex]", ...args);
}

interface UsageWindow {
  pct: number;
  reset?: number;
}

interface UsageResult {
  windows: Record<string, UsageWindow>;
  plan?: string;
  email?: string;
}

async function fetchUsage(apiKey: string): Promise<UsageResult | null> {
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

    const data = (await res.json()) as any;
    const windows: Record<string, UsageWindow> = {};
    const ONE_DAY_SECONDS = 86_400;

    for (const raw of [data.rate_limit?.primary_window, data.rate_limit?.secondary_window]) {
      if (!raw) continue;

      const durationSeconds: number = raw.limit_window_seconds ?? 0;
      const label =
        durationSeconds > 0 && durationSeconds < ONE_DAY_SECONDS
          ? "primary"
          : "week";

      windows[label] = {
        pct: raw.used_percent || 0,
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

function accountKey(slotIndex: number): string {
  return `${ACCOUNT_PREFIX}${slotIndex}`;
}

function getAccounts(authStorage: any): string[] {
  return ((authStorage.list?.() ?? []) as string[])
    .filter((k) => k.startsWith(ACCOUNT_PREFIX) && k !== ACTIVE_POINTER_KEY)
    .sort();
}

function isExpired(cred: any): boolean {
  const expires: unknown = cred?.expires;
  if (typeof expires !== "number") return true;
  return Date.now() >= expires;
}

async function resolveApiKey(
  authStorage: any,
  key: string,
  cred: any,
): Promise<string | undefined> {
  if (cred && !isExpired(cred) && typeof cred.access === "string") {
    return cred.access;
  }
  return authStorage.getApiKey(key).catch((err: unknown) => {
    debug("resolveApiKey error", key, err);
    return undefined;
  });
}

function getActive(authStorage: any): string | undefined {
  const pointer = authStorage.get(ACTIVE_POINTER_KEY) as any;
  const pointerKey: unknown =
    pointer?.type === "api_key" ? pointer.key : undefined;

  if (typeof pointerKey === "string" && authStorage.get(pointerKey)) {
    return pointerKey;
  }

  const legacyActive = authStorage.get(ACTIVE_KEY);
  if (!legacyActive) return undefined;

  for (const key of getAccounts(authStorage)) {
    const stored = authStorage.get(key);
    if (
      stored &&
      stored.type === legacyActive.type &&
      stored.access === legacyActive.access
    ) {
      return key;
    }
  }

  return undefined;
}

function setActive(authStorage: any, key: string, cred: any): void {
  authStorage.set(ACTIVE_KEY, cred);
  authStorage.set(ACTIVE_POINTER_KEY, { type: "api_key", key });
}

interface Row {
  key: string;
  index: number;
  email: string;
  plan?: string;
  windows: Array<{
    name: string;
    pct: number;
    reset?: number;
    color: string;
  }>;
  error?: string;
  active: boolean;
}

class List {
  private rows: Row[] = [];
  private loading = true;
  private selected = 0;
  private tui: { requestRender: () => void };
  private theme: any;
  private done: (result?: unknown) => void;
  private ctx: any;
  private busy = "";

  constructor(
    tui: any,
    theme: any,
    done: (result?: unknown) => void,
    ctx: any,
  ) {
    this.tui = tui;
    this.theme = theme;
    this.done = done;
    this.ctx = ctx;
    void this.init();
  }

  private dim(s: string): string {
    return this.theme.fg("muted", s);
  }

  private bold(s: string): string {
    return this.theme.bold(s);
  }

  private accent(s: string): string {
    return this.theme.fg("accent", s);
  }

  private async init(): Promise<void> {
    const authStorage = this.ctx.modelRegistry.authStorage;
    const accounts = getAccounts(authStorage);
    const activeKey = getActive(authStorage);

    const rows = await Promise.all(
      accounts.map(async (key): Promise<Row> => {
        const slotIndex = parseInt(key.slice(ACCOUNT_PREFIX.length), 10);
        const storedCred = authStorage.get(key);

        if (!storedCred) {
          return {
            key,
            index: slotIndex,
            email: "unknown",
            windows: [],
            error: "missing",
            active: key === activeKey,
          };
        }

        const apiKey = await resolveApiKey(authStorage, key, storedCred);

        if (!apiKey) {
          return {
            key,
            index: slotIndex,
            email: "unknown",
            windows: [],
            error: "auth expired",
            active: key === activeKey,
          };
        }

        const usage = await fetchUsage(apiKey);
        const email = parseEmailFromJwt(apiKey) ?? usage?.email;
        const windows: Row["windows"] = [];

        if (usage?.windows) {
          const sortOrder: Record<string, number> = { week: 0, primary: 1 };

          for (const [name, win] of Object.entries(usage.windows).sort(
            (a, b) => (sortOrder[a[0]] ?? 99) - (sortOrder[b[0]] ?? 99),
          )) {
            const remaining = 100 - win.pct;
            const color =
              remaining <= 10
                ? "error"
                : remaining <= 30
                  ? "warning"
                  : "success";

            windows.push({ name, pct: win.pct, reset: win.reset, color });
          }
        }

        return {
          key,
          index: slotIndex,
          email: email ?? "unknown",
          plan: usage?.plan,
          windows,
          error: usage ? undefined : "fetch failed",
          active: key === activeKey,
        };
      }),
    );

    this.rows = rows;

    const activeRowIdx = rows.findIndex((r) => r.active);
    if (activeRowIdx >= 0) this.selected = activeRowIdx;

    this.loading = false;
    this.tui.requestRender();
  }

  handleInput(ev: string): void {
    if (this.busy) return;

    if (matchesKey(ev, "escape")) {
      this.done();
      return;
    }

    if (matchesKey(ev, "up") || ev === "k") {
      this.selected = Math.max(0, this.selected - 1);
      this.tui.requestRender();
      return;
    }

    if (matchesKey(ev, "down") || ev === "j") {
      this.selected = Math.min(this.rows.length - 1, this.selected + 1);
      this.tui.requestRender();
      return;
    }

    if (matchesKey(ev, "enter")) {
      void this.withBusy("switch", () => this.doSwitch());
      return;
    }

    if (ev === "a") {
      void this.withBusy("add", () => this.doAdd());
      return;
    }

    if (matchesKey(ev, "backspace") || matchesKey(ev, "delete")) {
      void this.withBusy("remove", () => this.doRemove());
    }
  }

  private async withBusy(label: string, fn: () => Promise<void>): Promise<void> {
    this.busy = label;
    this.tui.requestRender();

    try {
      await fn();
    } catch (err) {
      debug("withBusy error", label, err);
      this.busy = "";
      this.tui.requestRender();
    }
  }

  private async doSwitch(): Promise<void> {
    const row = this.rows[this.selected];
    if (!row) return;

    const authStorage = this.ctx.modelRegistry.authStorage;
    const cred = authStorage.get(row.key);
    if (!cred) return;

    if (isExpired(cred)) {
      await resolveApiKey(authStorage, row.key, cred);
    }

    const freshCred = authStorage.get(row.key) ?? cred;
    setActive(authStorage, row.key, freshCred);
    this.done();
    await this.ctx.reload();
  }

  private async doAdd(): Promise<void> {
    try {
      const creds = await loginOpenAICodex({
        onAuth: ({ url, instructions }) => {
          this.ctx.ui.notify(`Open: ${url}`, "info");
          if (instructions) this.ctx.ui.notify(instructions, "info");

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
        onPrompt: async ({ message }) => {
          const v = await this.ctx.ui.input(message);
          if (!v?.trim()) throw new Error("Cancelled");
          return v.trim();
        },
        onProgress: (msg: string) => this.ctx.ui.notify(msg, "info"),
        originator: "pi",
      });

      const authStorage = this.ctx.modelRegistry.authStorage;
      let nextSlot = 0;

      for (const key of getAccounts(authStorage)) {
        const parsed = parseInt(key.slice(ACCOUNT_PREFIX.length), 10);
        if (!isNaN(parsed) && parsed >= nextSlot) nextSlot = parsed + 1;
      }

      const cred = { type: "oauth", ...creds };
      authStorage.set(accountKey(nextSlot), cred);
      setActive(authStorage, accountKey(nextSlot), cred);
      this.ctx.ui.notify(`Added & switched to [${nextSlot}]`, "success");
    } catch (e: any) {
      this.ctx.ui.notify(`Failed: ${e?.message || e}`, "error");
    }

    this.busy = "";
    this.loading = true;
    void this.init().then(() => this.tui.requestRender());
  }

  private async doRemove(): Promise<void> {
    const row = this.rows[this.selected];
    if (!row) return;

    const authStorage = this.ctx.modelRegistry.authStorage;

    if (row.active) {
      authStorage.remove(ACTIVE_KEY);
      authStorage.remove(ACTIVE_POINTER_KEY);
    }

    authStorage.remove(row.key);

    this.busy = "";
    this.loading = true;

    void this.init().then(() => {
      this.selected = Math.min(this.selected, Math.max(0, this.rows.length - 1));
      this.tui.requestRender();
    });
  }

  invalidate(): void {}

  dispose(): void {}

  render(width: number): string[] {
    const t = this.theme;
    const innerWidth = width - 4;
    const horizontal = "─".repeat(width - 2);

    const boxLine = (content: string): string => {
      const visibleLen = content.replace(/\x1b\[[0-9;]*m/g, "").length;
      return (
        this.dim("│ ") +
        content +
        " ".repeat(Math.max(0, innerWidth - visibleLen)) +
        this.dim(" │")
      );
    };

    const lines: string[] = [];

    if (this.busy) {
      lines.push(
        this.dim(`╭${horizontal}╮`),
        boxLine(this.bold(this.accent("multi-codex"))),
        this.dim(`├${horizontal}┤`),
        boxLine(`${this.busy}...`),
        this.dim(`╰${horizontal}╯`),
      );
      return lines;
    }

    lines.push(
      this.dim(`╭${horizontal}╮`),
      boxLine(this.bold(this.accent("multi-codex"))),
      this.dim(`├${horizontal}┤`),
    );

    if (this.loading) {
      lines.push(boxLine("loading..."));
    } else if (!this.rows.length) {
      lines.push(
        boxLine("no accounts"),
        boxLine(""),
        boxLine(this.dim("a  add account")),
      );
    } else {
      for (let i = 0; i < this.rows.length; i++) {
        const row = this.rows[i];

        const cursor = i === this.selected ? t.fg("accent", "▸ ") : "  ";
        const planLabel = row.plan ? t.fg("accent", ` ${row.plan}`) : "";
        const activeDot = row.active ? t.fg("success", " ●") : "";

        lines.push(
          boxLine(
            `${cursor}` +
              `${this.bold(`[${row.index}]`)} ${row.email}` +
              `${planLabel}${activeDot}`,
          ),
        );

        if (row.error) {
          lines.push(boxLine(this.dim(`   ${row.error}`)));
          continue;
        }

        for (const win of row.windows) {
          const filled = Math.min(10, Math.round(win.pct / 10));
          const empty = 10 - filled;
          const bar =
            t.fg(win.color, "█".repeat(filled)) + this.dim("░".repeat(empty));

          const pctStr = `${win.pct.toFixed(0).padStart(3)}%`;
          const resetStr = win.reset
            ? this.dim(` ${formatReset(new Date(win.reset))}`)
            : "";

          lines.push(
            boxLine(`   ${win.name.padEnd(8)} ${bar} ${pctStr}${resetStr}`),
          );
        }
      }
    }

    lines.push(
      this.dim(`├${horizontal}┤`),
      boxLine(this.dim("↑↓ select  a add  ↵ switch  ⌫ remove  esc close")),
      this.dim(`╰${horizontal}╯`),
    );

    return lines;
  }
}

function formatReset(date: Date): string {
  const diff = date.getTime() - Date.now();
  if (diff < 0) return "now";

  const minutes = Math.floor(diff / 60_000);
  if (minutes < 60) return `in ${minutes}m`;

  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    const remainderMin = minutes % 60;
    const suffix = remainderMin ? ` ${remainderMin}m` : "";
    return `in ${hours}h${suffix}`;
  }

  return `in ${Math.floor(hours / 24)}d`;
}

export default function (pi: ExtensionAPI) {
  pi.on("session_start", async (_event, ctx) => {
    const authStorage = ctx.modelRegistry.authStorage;

    for (const key of authStorage.list?.() ?? []) {
      if (key === ACTIVE_POINTER_KEY) continue;

      if (key.startsWith(ACCOUNT_PREFIX) || key === ACTIVE_KEY) {
        try {
          await authStorage.getApiKey(key);
        } catch (err) {
          debug("session_start refresh failed", key, err);
        }
      }
    }
  });

  pi.on("message_end", async (event, ctx) => {
    if (event.message.role !== "assistant") return;

    const msg = event.message as any;
    if (msg.stopReason !== "error") return;

    const errorMessage: string = msg.errorMessage ?? "";
    if (!/usage.limit/i.test(errorMessage)) return;

    const authStorage = ctx.modelRegistry.authStorage;
    const accounts = getAccounts(authStorage);
    if (accounts.length <= 1) return;

    const activeKey = getActive(authStorage);

    const scored = (
      await Promise.all(
        accounts.map(async (key) => {
          const cred = authStorage.get(key);
          if (!cred) return null;

          const apiKey = await resolveApiKey(authStorage, key, cred);
          if (!apiKey) return null;

          const usage = await fetchUsage(apiKey);
          if (!usage?.windows) return null;

          const now = Date.now();
          const primary = usage.windows["primary"];
          const week = usage.windows["week"];

          const primaryPct =
            primary?.reset && primary.reset <= now ? 0 : (primary?.pct ?? 0);
          const weekPct =
            week?.reset && week.reset <= now ? 0 : (week?.pct ?? 0);

          const usagePct = Math.max(primaryPct, weekPct);
          const resetAt = Math.min(
            primary?.reset ?? Infinity,
            week?.reset ?? Infinity,
          );

          return { key, pct: usagePct, reset: resetAt, cred: authStorage.get(key) ?? cred };
        }),
      )
    ).filter(
      (
        s,
      ): s is {
        key: string;
        pct: number;
        reset: number;
        cred: any;
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
      ctx.ui.notify("All accounts rate-limited — wait for reset", "error");
      return;
    }

    setActive(authStorage, best.key, best.cred);

    const resetDate = new Date(best.reset);
    ctx.ui.notify(
      `Switched to [${best.key.slice(ACCOUNT_PREFIX.length)}] ` +
        `(resets ${resetDate.getHours()}:` +
        `${String(resetDate.getMinutes()).padStart(2, "0")})`,
      "warning",
    );

    const entries = ctx.sessionManager.getEntries();

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
    handler: async (_args, ctx) => {
      await ctx.ui.custom(
        (
          tu: any,
          th: any,
          _kb: any,
          done: (result?: unknown) => void,
        ) => new List(tu, th, done, ctx),
      );
    },
  });
}

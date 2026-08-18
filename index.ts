import type {
  ApiKeyCredential,
  Credential,
  OAuthAuth,
  OAuthCredential,
} from '@earendil-works/pi-ai';
import { openaiCodexProvider } from '@earendil-works/pi-ai/providers/openai-codex';
import type {
  ExtensionAPI,
  ExtensionCommandContext,
  Theme,
} from '@earendil-works/pi-coding-agent';
import { AccountStore } from './auth-store.ts';
import { createInteraction } from './oauth-interaction.ts';
import {
  matchesKey,
  type TUI,
  type KeybindingsManager,
  type Component,
} from '@earendil-works/pi-tui';

const ACCOUNT_PREFIX = 'openai-codex-';
const ACTIVE_KEY = 'openai-codex';
const FETCH_TIMEOUT_MS = 10_000;

const DEBUG = !!process.env['MULTI_CODEX_DEBUG'];

function debug(...args: unknown[]): void {
  if (DEBUG) console.error('[multi-codex]', ...args);
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

interface UsageResult {
  windows: Record<string, { percent: number; reset?: number | undefined }>;
  plan?: string | undefined;
  email?: string | undefined;
}

async function fetchUsage(
  apiKey: string,
): Promise<UsageResult | 'unauthorized' | null> {
  try {
    const response = await fetch('https://chatgpt.com/backend-api/wham/usage', {
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'User-Agent': 'pi-agent',
        Accept: 'application/json',
      },
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
    });

    if (!response.ok) {
      debug('fetchUsage non-ok', response.status);
      if (response.status === 401 || response.status === 403)
        return 'unauthorized';
      return null;
    }

    const data = (await response.json()) as CodexUsageResponse;
    const windows: Record<
      string,
      { percent: number; reset?: number | undefined }
    > = {};
    const ONE_DAY_SECONDS = 86_400;

    for (const rawWindow of [
      data.rate_limit?.primary_window,
      data.rate_limit?.secondary_window,
    ]) {
      if (!rawWindow) continue;

      const durationSeconds: number = rawWindow.limit_window_seconds ?? 0;
      const label =
        durationSeconds > 0 && durationSeconds < ONE_DAY_SECONDS
          ? 'primary'
          : 'week';

      windows[label] = {
        percent: rawWindow.used_percent || 0,
        reset: rawWindow.reset_at ? rawWindow.reset_at * 1000 : undefined,
      };
    }

    let plan: string | undefined;
    if (data.plan_type) plan = data.plan_type;

    const email = typeof data.email === 'string' ? data.email : undefined;

    return { windows, plan, email };
  } catch (error) {
    debug('fetchUsage error', error);
    return null;
  }
}

function parseProfileFromJwt(apiKey: string): {
  email?: string | undefined;
  plan?: string | undefined;
} {
  try {
    const [, payloadB64] = apiKey.split('.');
    if (!payloadB64) return {};

    const decoded = JSON.parse(
      Buffer.from(payloadB64, 'base64url').toString('utf8'),
    );

    return {
      email: decoded?.['https://api.openai.com/profile']?.email as
        string | undefined,
      plan: decoded?.['https://api.openai.com/auth']?.chatgpt_plan_type as
        string | undefined,
    };
  } catch (error) {
    debug('parseProfileFromJwt error', error);
    return {};
  }
}

function makeAccountKey(index: number): string {
  return `${ACCOUNT_PREFIX}${index}`;
}

function codexOAuth(): OAuthAuth {
  const oauth = openaiCodexProvider().auth.oauth;
  if (!oauth) throw new Error('OpenAI Codex provider exposes no OAuth flow');
  return oauth;
}

function getAccountKeys(authStorage: AccountStore): string[] {
  return authStorage
    .list()
    .filter((key: string) => key.startsWith(ACCOUNT_PREFIX))
    .sort();
}

function getActiveAccountKey(authStorage: AccountStore): string | undefined {
  const activeCredential = authStorage.get(ACTIVE_KEY);
  if (!activeCredential) return undefined;

  for (const accountKey of getAccountKeys(authStorage)) {
    const credential = authStorage.get(accountKey);
    if (
      credential &&
      credential.type === activeCredential.type &&
      (activeCredential.type === 'oauth'
        ? (credential as OAuthCredential).access ===
          (activeCredential as OAuthCredential).access
        : (credential as ApiKeyCredential).key ===
          (activeCredential as ApiKeyCredential).key)
    ) {
      return accountKey;
    }
  }
  return undefined;
}

/**
 * Resolve a usable bearer token for an account.
 *
 * The stored credential already contains a token — use it directly. Do NOT call
 * authStorage.getApiKey(key) for numbered account keys: it looks up an OAuth
 * provider by the exact key string, but providers are only registered under
 * ACTIVE_KEY ("openai-codex"). Passing "openai-codex-0" hits
 * `getOAuthProvider(id) === undefined` and returns undefined, which this
 * extension renders as "auth expired" for EVERY account regardless of how
 * recently you logged in. That was the bug.
 *
 * We hand the raw token to the caller and let the actual usage request be the
 * source of truth — try the token first, refresh only if it genuinely fails.
 */
function getStoredToken(
  authStorage: AccountStore,
  key: string,
): string | undefined {
  const credential = authStorage.get(key);
  if (!credential) return undefined;
  if (credential.type === 'api_key') return credential.key;
  if (credential.type === 'oauth')
    return (credential as OAuthCredential).access;
  return undefined;
}

/**
 * Refresh an expired numbered account directly through the provider OAuth flow,
 * persisting the refreshed credential back to the account's own storage key.
 *
 * The flow refreshes a credential we pass in, so numbered accounts no longer
 * need routing through ACTIVE_KEY. ACTIVE_KEY only tracks the selected account.
 */
async function refreshAccount(
  authStorage: AccountStore,
  key: string,
): Promise<string | undefined> {
  const original = authStorage.get(key);
  if (!original || original.type !== 'oauth') return undefined;

  const wasActive = getActiveAccountKey(authStorage) === key;

  const refreshed = await codexOAuth()
    .refresh(original, AbortSignal.timeout(FETCH_TIMEOUT_MS))
    .catch((error: unknown) => {
      debug('refresh error', key, error);
      return undefined;
    });
  if (!refreshed) return undefined;

  authStorage.set(key, refreshed);
  if (wasActive) authStorage.set(ACTIVE_KEY, refreshed);
  return refreshed.access;
}

type UsageColor = 'error' | 'warning' | 'success';

interface UsageWindowRow {
  name: string;
  percent: number;
  reset?: number | undefined;
  color: UsageColor;
}

interface AccountRow {
  key: string;
  index: number;
  email: string;
  plan?: string | undefined;
  usageWindows: UsageWindowRow[];
  error?: string | undefined;
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
  private busy = '';

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
    return this.theme.fg('muted', text);
  }

  private bold(text: string): string {
    return this.theme.bold(text);
  }

  private accent(text: string): string {
    return this.theme.fg('accent', text);
  }

  private async init() {
    const authStorage = new AccountStore();
    const accountKeys = getAccountKeys(authStorage);
    const activeKey = getActiveAccountKey(authStorage);
    const rows: AccountRow[] = [];

    for (const accountKey of accountKeys) {
      const accountIndex = parseInt(
        accountKey.slice(ACCOUNT_PREFIX.length),
        10,
      );
      const credential = authStorage.get(accountKey);
      if (!credential) {
        rows.push({
          key: accountKey,
          index: accountIndex,
          email: 'unknown',
          usageWindows: [],
          error: 'missing',
          active: accountKey === activeKey,
        });
        continue;
      }

      let apiKey = getStoredToken(authStorage, accountKey);
      let usage = apiKey ? await fetchUsage(apiKey) : 'unauthorized';

      if (usage === 'unauthorized') {
        const refreshed = await refreshAccount(authStorage, accountKey);
        if (refreshed) {
          apiKey = refreshed;
          usage = await fetchUsage(refreshed);
        }
      }

      if (!apiKey || usage === 'unauthorized') {
        rows.push({
          key: accountKey,
          index: accountIndex,
          email: 'unknown',
          usageWindows: [],
          error: 'auth expired',
          active: accountKey === activeKey,
        });
        continue;
      }

      const usageData = usage; // null = fetch failed, UsageResult = ok
      const jwtProfile = parseProfileFromJwt(apiKey);
      const email = jwtProfile.email ?? usageData?.email;
      const plan = usageData?.plan ?? jwtProfile.plan;
      const usageWindows: UsageWindowRow[] = [];

      if (usageData?.windows) {
        const entries = Object.entries(usageData.windows).sort(
          (a, b) =>
            (USAGE_SORT_ORDER[a[0]] ?? 99) - (USAGE_SORT_ORDER[b[0]] ?? 99),
        );
        for (const [windowName, windowData] of entries) {
          const remaining = 100 - windowData.percent;
          usageWindows.push({
            name: windowName,
            percent: windowData.percent,
            reset: windowData.reset,
            color:
              remaining <= 10
                ? 'error'
                : remaining <= 30
                  ? 'warning'
                  : 'success',
          });
        }
      }

      rows.push({
        key: accountKey,
        index: accountIndex,
        email: email ?? 'unknown',
        plan,
        usageWindows,
        error: usageData ? undefined : 'fetch failed',
        active: accountKey === activeKey,
      });
    }

    rows.sort(compareRowsByWeeklyReset);
    this.rows = rows;

    const activeIndex = rows.findIndex((row) => row.active);
    if (activeIndex >= 0) this.selectedIndex = activeIndex;

    this.loading = false;
    this.tui.requestRender();
  }

  handleInput(event: string): void {
    if (this.busy) return;

    if (matchesKey(event, 'escape')) {
      this.done(undefined);
      return;
    }

    if (matchesKey(event, 'up') || event === 'k') {
      this.selectedIndex = Math.max(0, this.selectedIndex - 1);
      this.tui.requestRender();
      return;
    }

    if (matchesKey(event, 'down') || event === 'j') {
      this.selectedIndex = Math.min(
        this.rows.length - 1,
        this.selectedIndex + 1,
      );
      this.tui.requestRender();
      return;
    }

    if (matchesKey(event, 'enter')) {
      void this.withBusy('switch', () => this.switchAccount());
      return;
    }

    if (event === 'a') {
      void this.withBusy('add', () => this.addAccount());
      return;
    }

    if (matchesKey(event, 'backspace') || matchesKey(event, 'delete')) {
      void this.withBusy('remove', () => this.removeAccount());
    }
  }

  private async withBusy(label: string, action: () => Promise<void>) {
    this.busy = label;
    this.tui.requestRender();
    try {
      await action();
    } finally {
      this.busy = '';
      this.tui.requestRender();
    }
  }

  private async switchAccount() {
    const row = this.rows[this.selectedIndex];
    if (!row || row.active) return;

    const authStorage = new AccountStore();
    // Copy this account's stored credential into the active provider key without refreshing ACTIVE_KEY.
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
        'error',
      );
    }
  }

  private async addAccount() {
    try {
      const interaction = createInteraction(
        {
          notify: (message, level) => this.context.ui.notify(message, level),
          input: (message) => this.context.ui.input(message),
        },
        new AbortController().signal,
      );
      const credentials = await codexOAuth().login(interaction);

      const authStorage = new AccountStore();

      let nextIndex = 0;
      for (const key of authStorage.list()) {
        if (key.startsWith(ACCOUNT_PREFIX)) {
          const existingIndex = parseInt(key.slice(ACCOUNT_PREFIX.length), 10);
          if (!isNaN(existingIndex) && existingIndex >= nextIndex) {
            nextIndex = existingIndex + 1;
          }
        }
      }

      const credential: Credential = credentials;
      authStorage.set(makeAccountKey(nextIndex), credential);
      authStorage.set(ACTIVE_KEY, credential);

      this.context.ui.notify('Added & switched account', 'info');
    } catch (error) {
      this.context.ui.notify(
        `Failed: ${error instanceof Error ? error.message : String(error)}`,
        'error',
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

    const authStorage = new AccountStore();
    if (row.active) {
      authStorage.remove(ACTIVE_KEY);
    }
    authStorage.remove(row.key);

    this.loading = true;
    void this.init().then(() => {
      this.selectedIndex = Math.max(
        0,
        Math.min(this.selectedIndex, this.rows.length - 1),
      );
      this.tui.requestRender();
    });
  }

  invalidate(): void {}
  dispose(): void {}

  render(width: number): string[] {
    const theme = this.theme;
    const innerWidth = width - 4;
    const horizontalLine = '─'.repeat(width - 2);

    const boxLine = (content: string): string => {
      const visible = content.replace(/\x1b\[[0-9;]*m/g, '');
      const padding = ' '.repeat(Math.max(0, innerWidth - visible.length));
      return this.dim('│ ') + content + padding + this.dim(' │');
    };

    const lines: string[] = [];

    if (this.busy) {
      lines.push(
        this.dim(`╭${horizontalLine}╮`),
        boxLine(this.bold(this.accent('multi-codex'))),
        this.dim(`├${horizontalLine}┤`),
        boxLine(`${this.busy}...`),
        this.dim(`╰${horizontalLine}╯`),
      );
      return lines;
    }

    lines.push(
      this.dim(`╭${horizontalLine}╮`),
      boxLine(this.bold(this.accent('multi-codex'))),
      this.dim(`├${horizontalLine}┤`),
    );

    if (this.loading) {
      lines.push(boxLine('loading...'));
    } else if (!this.rows.length) {
      lines.push(
        boxLine('no accounts'),
        boxLine(''),
        boxLine(this.dim('a  add account')),
      );
    } else {
      for (const [index, row] of this.rows.entries()) {
        const isSelected = index === this.selectedIndex;

        const planLabel = row.plan ? theme.fg('accent', ` ${row.plan}`) : '';
        const cursor = isSelected ? theme.fg('accent', '▸ ') : '  ';
        const activeDot = row.active ? theme.fg('success', ' ●') : '';
        lines.push(
          boxLine(`${cursor}${this.bold(row.email)}${planLabel}${activeDot}`),
        );

        if (row.error) {
          lines.push(boxLine(this.dim(`   ${row.error}`)));
          continue;
        }

        for (const usageWindow of row.usageWindows) {
          const filled = Math.min(10, Math.round(usageWindow.percent / 10));
          const empty = 10 - filled;
          const bar =
            theme.fg(usageWindow.color, '█'.repeat(filled)) +
            this.dim('░'.repeat(empty));
          const resetLabel = usageWindow.reset
            ? this.dim(` ${formatCountdown(new Date(usageWindow.reset))}`)
            : '';

          lines.push(
            boxLine(
              `   ${usageWindow.name.padEnd(7)} ${bar} ${usageWindow.percent.toFixed(0).padStart(3)}%${resetLabel}`,
            ),
          );
        }
      }
    }

    lines.push(
      this.dim(`├${horizontalLine}┤`),
      boxLine(this.dim('↑↓ select  a add  ↵ switch  ⌫ remove  esc close')),
      this.dim(`╰${horizontalLine}╯`),
    );

    return lines;
  }
}

function getWeeklyReset(row: AccountRow): number {
  return (
    row.usageWindows.find((window) => window.name === 'week')?.reset ??
    Number.POSITIVE_INFINITY
  );
}

function compareRowsByWeeklyReset(
  first: AccountRow,
  second: AccountRow,
): number {
  const resetDiff = getWeeklyReset(first) - getWeeklyReset(second);
  if (resetDiff !== 0) return resetDiff;
  return first.email.localeCompare(second.email);
}

function formatCountdown(date: Date): string {
  const diffMs = date.getTime() - Date.now();
  if (diffMs < 0) return 'now';

  const minutes = Math.floor(diffMs / 60_000);
  if (minutes < 60) return `in ${minutes}m`;

  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  const minuteSuffix = remainingMinutes ? ` ${remainingMinutes}m` : '';

  if (hours < 24) return `in ${hours}h${minuteSuffix}`;

  return `in ${Math.floor(hours / 24)}d`;
}

export default function (pi: ExtensionAPI) {
  pi.on('message_end', async (event, context) => {
    const message = event.message;
    if (message.role !== 'assistant') return;
    if (message.stopReason !== 'error') return;

    const errorMessage: string = message.errorMessage ?? '';
    if (!/usage.limit/i.test(errorMessage)) return;

    const authStorage = new AccountStore();
    const accountKeys = getAccountKeys(authStorage);
    if (accountKeys.length <= 1) return;

    const activeKey = getActiveAccountKey(authStorage);

    const scored = (
      await Promise.all(
        accountKeys.map(async (key) => {
          let accessToken = getStoredToken(authStorage, key);
          let usage = accessToken
            ? await fetchUsage(accessToken)
            : 'unauthorized';
          if (usage === 'unauthorized') {
            const refreshed = await refreshAccount(authStorage, key);
            if (refreshed) {
              accessToken = refreshed;
              usage = await fetchUsage(refreshed);
            }
          }
          if (
            !accessToken ||
            usage === 'unauthorized' ||
            !usage ||
            !usage.windows
          )
            return null;
          const usageData = usage;

          const now = Date.now();
          const primary = usageData.windows['primary'];
          const week = usageData.windows['week'];

          const primaryPercent =
            primary?.reset && primary.reset <= now
              ? 0
              : (primary?.percent ?? 0);
          const weekPercent =
            week?.reset && week.reset <= now ? 0 : (week?.percent ?? 0);

          const usagePercent = Math.max(primaryPercent, weekPercent);
          const resetAt = Math.min(
            primary?.reset ?? Infinity,
            week?.reset ?? Infinity,
          );

          const credential = authStorage.get(key);
          if (!credential) return null;
          return { key, percent: usagePercent, reset: resetAt, credential };
        }),
      )
    ).filter(
      (
        entry,
      ): entry is {
        key: string;
        percent: number;
        reset: number;
        credential: Credential;
      } => entry !== null,
    );

    if (scored.length === 0) return;

    scored.sort((first, second) => {
      const firstExhausted = first.percent >= 100 ? 1 : 0;
      const secondExhausted = second.percent >= 100 ? 1 : 0;
      if (firstExhausted !== secondExhausted)
        return firstExhausted - secondExhausted;
      return first.reset - second.reset;
    });

    const best = scored.find((entry) => entry.key !== activeKey) ?? scored[0];
    if (!best) return;

    if (best.percent >= 100) {
      const soonestReset = scored
        .map((entry) => entry.reset)
        .filter((reset) => Number.isFinite(reset))
        .sort((first, second) => first - second)[0];
      const resetMessage = soonestReset
        ? ` — soonest reset ${formatCountdown(new Date(soonestReset))}`
        : '';
      context.ui.notify(`All accounts rate-limited${resetMessage}`, 'error');
      return;
    }

    authStorage.set(ACTIVE_KEY, best.credential);

    context.ui.notify(
      `Switched account (resets ${formatCountdown(new Date(best.reset))})`,
      'warning',
    );

    const entries = context.sessionManager.getEntries();

    for (const entry of [...entries].reverse()) {
      if (entry.type === 'message' && entry.message.role === 'user') {
        pi.sendUserMessage(entry.message.content, { deliverAs: 'followUp' });
        break;
      }
    }
  });

  pi.registerCommand('multi-codex', {
    description: 'Manage multiple OpenAI Codex accounts',
    handler: async (_args, context) => {
      await context.ui.custom(
        (tui, theme, keybindings, done) =>
          new AccountList(tui, theme, keybindings, done, context),
      );
    },
  });
}

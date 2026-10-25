import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import extension from '../index.ts';

type Handler = (event: unknown, context: unknown) => Promise<void>;

function setUp(auth: Record<string, unknown>, usage: unknown) {
  const dir = mkdtempSync(join(tmpdir(), 'multi-codex-'));
  writeFileSync(join(dir, 'auth.json'), JSON.stringify(auth));
  process.env['PI_CODING_AGENT_DIR'] = dir;
  globalThis.fetch = async () => new Response(JSON.stringify(usage), { status: 200 });

  const handlers: Record<string, Handler> = {};
  extension({
    on: (name: string, handler: Handler) => {
      handlers[name] = handler;
    },
    registerCommand: () => {},
    sendUserMessage: () => {},
  } as never);

  const notifications: string[] = [];
  const context = {
    ui: { notify: (message: string) => notifications.push(message) },
    sessionManager: { getEntries: () => [] },
  };
  const event = {
    message: { role: 'assistant', stopReason: 'error', errorMessage: 'usage limit reached' },
  };
  return { run: () => handlers['message_end']!(event, context), notifications };
}

test('switch notice omits the reset time when the usage has none', async () => {
  const first = { type: 'api_key', key: 'first' };
  const second = { type: 'api_key', key: 'second' };
  const { run, notifications } = setUp(
    { 'openai-codex': first, 'openai-codex-0': first, 'openai-codex-1': second },
    { rate_limit: { primary_window: { limit_window_seconds: 18_000, used_percent: 5 } } },
  );

  await run();

  assert.deepEqual(notifications, ['Switched account']);
});

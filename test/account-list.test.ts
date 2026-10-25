import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import extension from '../index.ts';

type Command = { handler: (args: string, context: unknown) => Promise<void> };
type Rendered = { render(width: number): string[] };

async function openList(auth: Record<string, unknown>, emails: Record<string, string>) {
  const dir = mkdtempSync(join(tmpdir(), 'multi-codex-'));
  writeFileSync(join(dir, 'auth.json'), JSON.stringify(auth));
  process.env['PI_CODING_AGENT_DIR'] = dir;
  globalThis.fetch = async (_url, init) => {
    const token = String(new Headers(init?.headers).get('Authorization')).replace('Bearer ', '');
    return new Response(JSON.stringify({ email: emails[token] }), { status: 200 });
  };

  let command: Command | undefined;
  extension({
    on: () => {},
    registerCommand: (_name: string, options: Command) => {
      command = options;
    },
  } as never);

  let component: Rendered | undefined;
  const theme = { fg: (_color: string, text: string) => text, bold: (text: string) => text };
  const context = {
    ui: {
      custom: (factory: (...args: unknown[]) => Rendered) => {
        component = factory({ requestRender: () => {} }, theme, {}, () => {});
        return new Promise(() => {});
      },
    },
  };
  void command!.handler('', context);

  for (let attempt = 0; attempt < 100; attempt++) {
    const lines = component!.render(80);
    if (!lines.some((line) => line.includes('loading...'))) return lines;
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  throw new Error('account list did not load');
}

test('accounts without a weekly reset are listed by email', async () => {
  const lines = await openList(
    {
      'openai-codex-0': { type: 'api_key', key: 'zed' },
      'openai-codex-1': { type: 'api_key', key: 'amy' },
    },
    { zed: 'zed@example.com', amy: 'amy@example.com' },
  );

  const order = lines.filter((line) => line.includes('@example.com')).map((line) => line.trim());
  assert.match(order[0] ?? '', /amy@example\.com/);
  assert.match(order[1] ?? '', /zed@example\.com/);
});

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createInteraction } from '../src/oauth-interaction.ts';

test('manual code prompt is dismissed when its signal aborts', async () => {
  const interaction = createInteraction(
    {
      notify: () => {},
      input: (_message, signal) =>
        new Promise((resolve) => signal?.addEventListener('abort', () => resolve(undefined))),
    },
    new AbortController().signal,
  );
  const controller = new AbortController();
  const answer = interaction.prompt({
    type: 'manual_code',
    message: 'Paste code',
    signal: controller.signal,
  });
  controller.abort();
  const timeout = new Promise((resolve) => setTimeout(() => resolve('pending'), 100));
  const outcome = await Promise.race([
    answer.then(
      () => 'resolved',
      () => 'rejected',
    ),
    timeout,
  ]);
  assert.equal(outcome, 'rejected');
});

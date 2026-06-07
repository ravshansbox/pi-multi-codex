import assert from "node:assert/strict";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { test } from "node:test";
import ts from "typescript";

async function loadExtension() {
	const source = await readFile(new URL("../index.ts", import.meta.url), "utf8");
	const output = ts.transpileModule(source, {
		compilerOptions: {
			module: ts.ModuleKind.ESNext,
			target: ts.ScriptTarget.ES2022,
		},
	}).outputText;
	const directory = new URL("../.tmp-test/", import.meta.url);
	await rm(directory, { recursive: true, force: true });
	await mkdir(directory, { recursive: true });
	const moduleUrl = new URL(`index-${Date.now()}.mjs`, directory);
	await writeFile(moduleUrl, output);
	const loaded = await import(moduleUrl.href);
	await rm(directory, { recursive: true, force: true });
	return loaded;
}

class AuthStorage {
	#values = new Map();

	list() {
		return [...this.#values.keys()];
	}

	get(key) {
		return this.#values.get(key);
	}

	set(key, value) {
		this.#values.set(key, value);
	}

	remove(key) {
		this.#values.delete(key);
	}
}

test("auto-switch reset warning keeps date context for monthly resets", async () => {
	const resetAt = Date.now() + 30 * 24 * 60 * 60 * 1000;
	const previousFetch = globalThis.fetch;
	globalThis.fetch = async () => ({
		ok: true,
		json: async () => ({
			rate_limit: {
				primary_window: {
					limit_window_seconds: 30 * 24 * 60 * 60,
					used_percent: 10,
					reset_at: Math.floor(resetAt / 1000),
				},
			},
		}),
	});

	try {
		const authStorage = new AuthStorage();
		const activeCredential = { type: "oauth", access: "active" };
		const freeCredential = { type: "oauth", access: "free" };
		authStorage.set("openai-codex-0", activeCredential);
		authStorage.set("openai-codex-1", freeCredential);
		authStorage.set("openai-codex", activeCredential);

		const notifications = [];
		let messageEnd;
		const pi = {
			on(event, handler) {
				if (event === "message_end") messageEnd = handler;
			},
			sendUserMessage() {},
			registerCommand() {},
		};
		const context = {
			modelRegistry: { authStorage },
			ui: {
				notify(message, severity) {
					notifications.push({ message, severity });
				},
			},
			sessionManager: {
				getEntries() {
					return [];
				},
			},
		};
		const extension = await loadExtension();
		extension.default(pi);

		await messageEnd({ message: { role: "assistant", stopReason: "error", errorMessage: "usage.limit" } }, context);

		assert.equal(notifications.length, 1);
		assert.equal(notifications[0].severity, "warning");
		assert.match(notifications[0].message, /^Switched account \(resets in \d+d\)$/);
	} finally {
		globalThis.fetch = previousFetch;
	}
});

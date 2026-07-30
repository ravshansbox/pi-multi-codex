# pi-multi-codex

Multi-account OpenAI Codex routing extension for pi.

## Install

```json
{
  "extensions": ["github:ravshansbox/pi-multi-codex"]
}
```

## Usage

Pi loads the extension from `./src/index.ts`. Run `/multi-codex` in the TUI to add Codex accounts, switch between them, or remove old ones.

For example, if the active Codex account hits a usage limit during a session, the extension can switch to another saved account and resend the latest user message as a follow-up.

## Development

```bash
npm install
npm run typecheck
```

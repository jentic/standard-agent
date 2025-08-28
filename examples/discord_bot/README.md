## Discord Bot Example — Standard Agent

This example lets you converse with a Standard Agent from Discord using mention-gated behavior.

## Quick Start

From the project root:

```bash
pip install -r examples/discord_bot/requirements.txt
python examples/discord_bot/bot.py
```

In Discord:
- Invite the bot to your server (see “Create a Discord App” below)
- Mention the bot in a channel it can read: `@your-bot <goal>`
- Optional: restrict to specific channels via `DISCORD_ALLOWED_CHANNEL_IDS`

### Commands

- `@your-bot reasoner list` — show available strategies and the current one
- `@your-bot reasoner <react|rewoo>` — switch reasoning strategy (default: rewoo)
- `@your-bot configure <JENTIC_AGENT_API_KEY>` — set the Agent API key at runtime
  - Note: sharing secrets in-channel is unsafe; prefer environment variables
- `@your-bot kill` — clear the API key and reset the agent

## Create a Discord App

1. Create the app
   - Open `https://discord.com/developers/applications` → New Application → name your app.

2. Add a Bot and get the Bot Token
   - Left sidebar → Bot → Add Bot → Reset Token and copy it.
   - Save as `DISCORD_BOT_TOKEN` in your `.env`.

3. Enable Privileged Intents
   - Left sidebar → Bot → Privileged Gateway Intents → enable “Message Content Intent”.

4. Invite the bot to your server
   - Left sidebar → OAuth2 → URL Generator → Scopes: `bot`
   - Bot Permissions: `View Channels`, `Read Message History`, `Send Messages`
   - Use the generated URL to invite the bot to your server.

## Environment Variables

Add the following to your `.env` in the project root:

- `DISCORD_BOT_TOKEN` — Bot token for logging in
- `DISCORD_ALLOWED_CHANNEL_IDS` (optional) — comma-separated allowlist of channel IDs

## Usage

Mention the bot in a channel it can read, for example:

```
@your-bot find nyt articles on OpenAI
```

The bot treats the message (minus the mention) as the goal, runs the agent, and replies in the same channel. Long answers are split into 2000‑character chunks.

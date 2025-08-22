# Slack Bot Example — Standard Agent

This example lets you converse with a Standard Agent from Slack using Socket Mode.

## Quick Start

From the project root:

```bash
pip install -r examples/slack/requirements.txt
python examples/slack/slack_agent.py
```

In Slack:
- Make sure the app is installed to your workspace (App settings → OAuth & Permissions → Install to Workspace)
- Invite the bot: `/invite @your-bot`
- Configure the agent key: `/standard-agent configure` (paste API key from app.jentic.com)
- Talk to the agent: `@your-bot <goal>` or DM the bot

### Slash commands

- `/standard-agent configure` — open a modal to paste your Jentic Agent API Key
- `/standard-agent reasoner <react|rewoo>` — switch reasoning strategy (default: rewoo)
- `/standard-agent reasoner list` — show available strategies and the current one
- `/standard-agent kill` — clear the API key and reset the agent

## Create a Slack App

1. Create the app
   - Open `https://api.slack.com/apps` → Create New App → From scratch → name your app and pick your workspace.

2. Enable Socket Mode and get the App Token
   - Left sidebar → Features → Socket Mode → toggle On.
   - Click “Generate App-Level Token”, add scope `connections:write`, create it, and copy the token (starts with `xapp-`).
   - Save it as `SLACK_APP_TOKEN` in your `.env`.

3. Add bot scopes and install the app
   - Left sidebar → Features → OAuth & Permissions → Scopes → Bot Token Scopes: add
     - `commands` (for slash commands)
     - `chat:write`
     - `app_mentions:read`
     - `im:history`
   - Click “Install to Workspace” (or “Reinstall to Workspace”).
   - Copy the “Bot User OAuth Token” (starts with `xoxb-`) and save it as `SLACK_BOT_TOKEN` in your `.env`.

4. Get the Signing Secret
   - Left sidebar → Basic Information → App Credentials → “Signing Secret” → Show and copy.
   - Save it as `SLACK_SIGNING_SECRET` in your `.env` (recommended even in Socket Mode).

5. Subscribe to events (for messages)
   - Left sidebar → Features → Event Subscriptions → toggle On.
   - Under “Subscribe to bot events”, add:
     - `app_mention` (channel mentions)
     - `message.im` (direct messages)
   - Save changes. With Socket Mode enabled, no public URL is required.

6. Create slash command and enable interactivity
   - Left sidebar → Features → Slash Commands → Create New Command.
   - Command: `/standard-agent`
   - Request URL: (leave blank for Socket Mode)
   - Save.
   - Left sidebar → Interactivity & Shortcuts → toggle On (no URL needed for Socket Mode).


## Environment Variables

Add the following to your `.env` in the project root:

- `SLACK_APP_TOKEN` — App-level token with `connections:write` (Socket Mode)
- `SLACK_BOT_TOKEN` — Bot token for posting messages
- `SLACK_SIGNING_SECRET` — Signing secret (not strictly required for Socket Mode, but recommended)

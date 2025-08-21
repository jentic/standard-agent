# Slack Bot Example — Standard Agent

This example lets you converse with a Standard Agent from Slack using Socket Mode.

## Prerequisites

- A Slack workspace where you can install custom apps
- Python environment with project dependencies installed
- .env configured with an LLM provider key and (optionally) tool provider key

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

7. Invite and test
   - In Slack, invite your bot to a channel: `/invite @your-bot`.
   - Configure the agent key via slash command: `/standard-agent configure` (a modal opens).
   - Optionally pick a reasoner profile: `/standard-agent reasoner-profile react` or `rewoo`.
   - Mention the bot in a channel: `@your-bot find articles about AI` (the bot replies in a thread).
   - Or DM the bot directly and type your goal.

## Environment Variables

Add the following to your `.env` in the project root:

- `SLACK_APP_TOKEN` — App-level token with `connections:write` (Socket Mode)
- `SLACK_BOT_TOKEN` — Bot token for posting messages
- `SLACK_SIGNING_SECRET` — Signing secret (not strictly required for Socket Mode, but recommended)


## Install Dependencies

Install core deps, then the Slack example deps:
```bash
pip install -r examples/slack/requirements.txt
```

## Run

From the project root:

```bash
python examples/slack/slack_agent.py
```

- Configure: `/standard-agent configure` (paste Jentic Agent API Key)
- Switch profile: `/standard-agent reasoner-profile <react|rewoo>`
- Mention-based: `@your-bot find the latest news about AI`
- DM-based: send a goal as a direct message to the bot

### Quick flow

1. Create a Slack channel.
2. Invite the bot: `/invite @your-bot`.
3. If no JENTIC_AGENT_API_KEY key in `.env`, run `/standard-agent configure` and paste your Agent API Key which you can get from app.jentic.com.
4. (Optional) Set profile: `/standard-agent reasoner-profile <react|rewoo>`. default is `rewoo`.
5. Talk to the agent with `@your-bot <goal>` or DM the bot.

## Notes

- The agent composes LLM + Tools + Memory + Reasoner. Choose the profile via `/standard-agent reasoner-profile <react|rewoo>`.
- For tool usage, configure credentials for the tools you intend to use; default provider is the Jentic catalog when `JENTIC_AGENT_API_KEY` is present.
- Messages are answered synchronously; long-running tool calls will block until completion.



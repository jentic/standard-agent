### Discord Bot (mention-gated)

- Responds only when the bot is mentioned (e.g., @Standard_Agent) or in DMs.
- Uses each mentioned message (minus the mention text) as the agent goal and replies in the same channel.

#### Setup
- In `.env` at project root, set:
  - `DISCORD_BOT_TOKEN=...`
  - `LLM_MODEL=claude-sonnet-4` (or your choice)
  - At least one LLM provider key supported by `litellm` (e.g., OpenAI/Anthropic/Google)
  - Optional: `DISCORD_ALLOWED_CHANNEL_IDS=123456789012345678,987654321098765432`
  - Optional: `JENTIC_AGENT_API_KEY=...` to enable external tools

- Discord Developer Portal:
  - Enable Message Content Intent for your bot application
  - Invite bot with scopes: `bot`
  - Permissions: View Channels, Read Message History, Send Messages

#### Install and run
```bash
make install
python examples/discord_bot/bot.py
```

Mention the bot in a channel it can read, for example:
```
@Standard_Agent summarize https://example.com/article
```

The bot will treat the message (minus the mention) as the goal, run the agent, and post the final answer back to the same channel. Long answers are split into 2000‑char chunks.



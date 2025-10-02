# Microsoft Teams Bot Example — Standard Agent

This example lets you converse with a Standard Agent from Microsoft Teams using the Bot Framework.

## Quick Start

From the project root:

```bash
pip install -r examples/teams/requirements.txt
python examples/teams/teams_agent.py
```

In Microsoft Teams:
- Add the bot to your team or chat (see "Create a Teams App" below)
- Mention the bot in a channel: `@StandardAgent <goal>`
- Use slash commands to configure and manage the agent

### Available Commands

- `/configure` — Instructions for setting up the Agent API Key
- `/reasoner` — List available reasoning strategies and current one
- `/reasoner <react|rewoo>` — Switch reasoning strategy (default: rewoo)
- `/kill` — Clear the API key and reset the agent
- `/help` — Show available commands and usage

## Create a Microsoft Teams App

### Prerequisites

1. A Microsoft 365 account with Teams access
2. Admin permissions to install apps in your Teams environment (or developer tenant)
3. Access to [Azure Portal](https://portal.azure.com) for app registration

### Step 1: Register Your Bot in Azure

1. **Create an Azure Bot Resource**
   - Go to [Azure Portal](https://portal.azure.com)
   - Create a new resource → Search for "Azure Bot" → Create
   - Fill in the required details:
     - Bot handle: Choose a unique name (e.g., `standard-agent-bot`)
     - Subscription: Your Azure subscription
     - Resource group: Create new or use existing
     - Pricing tier: Free (F0) for development
     - Microsoft App ID: Select "Create new Microsoft App ID"

2. **Get Your App Credentials**
   - After creation, go to your bot resource
   - Navigate to "Configuration" in the left sidebar
   - Copy the "Microsoft App ID" - save as `MICROSOFT_APP_ID` in your `.env`
   - Click "Manage" next to the Microsoft App ID to go to App Registration

3. **Create App Password**
   - In the App Registration page, go to "Certificates & secrets"
   - Click "New client secret"
   - Add a description and set expiration
   - Copy the secret value immediately (you won't see it again)
   - Save as `MICROSOFT_APP_PASSWORD` in your `.env`

### Step 2: Configure Bot Endpoint

1. **Set the Messaging Endpoint**
   - Back in your Azure Bot resource, go to "Configuration"
   - Set the "Messaging endpoint" to your bot's URL:
     - For local development: `https://your-ngrok-url.ngrok.io/api/messages`
     - For Azure deployment: `https://your-app-name.azurewebsites.net/api/messages`
   - Save the configuration

### Step 3: Enable Teams Channel

1. **Add Microsoft Teams Channel**
   - In your Azure Bot resource, go to "Channels"
   - Click on the Microsoft Teams icon
   - Click "Apply" to enable the Teams channel
   - Once enabled, you can test your bot or get the teams app package

### Step 4: Create Teams App Manifest

Create a `manifest.json` file with your bot details:

```json
{
  "$schema": "https://developer.microsoft.com/en-us/json-schemas/teams/v1.16/MicrosoftTeams.schema.json",
  "manifestVersion": "1.16",
  "version": "1.0.0",
  "id": "YOUR_MICROSOFT_APP_ID",
  "packageName": "com.standardagent.teamsbot",
  "developer": {
    "name": "Standard Agent",
    "websiteUrl": "https://github.com/striver-24/standard-agent",
    "privacyUrl": "https://github.com/striver-24/standard-agent",
    "termsOfUseUrl": "https://github.com/striver-24/standard-agent"
  },
  "icons": {
    "color": "color.png",
    "outline": "outline.png"
  },
  "name": {
    "short": "Standard Agent",
    "full": "Standard Agent Bot"
  },
  "description": {
    "short": "AI-powered reasoning agent for Teams",
    "full": "Standard Agent provides AI-powered reasoning capabilities directly in Microsoft Teams using ReACT and ReWOO methodologies."
  },
  "accentColor": "#FFFFFF",
  "bots": [
    {
      "botId": "YOUR_MICROSOFT_APP_ID",
      "scopes": [
        "personal",
        "team",
        "groupchat"
      ],
      "supportsFiles": false,
      "isNotificationOnly": false,
      "commandLists": [
        {
          "scopes": [
            "personal",
            "team",
            "groupchat"
          ],
          "commands": [
            {
              "title": "help",
              "description": "Show available commands and usage"
            },
            {
              "title": "configure",
              "description": "Configure the Agent API Key"
            },
            {
              "title": "reasoner",
              "description": "Switch or list reasoning strategies"
            },
            {
              "title": "kill",
              "description": "Reset the agent and clear API key"
            }
          ]
        }
      ]
    }
  ],
  "permissions": [
    "identity",
    "messageTeamMembers"
  ],
  "validDomains": []
}
```

### Step 5: Install the Bot in Teams

1. **Create App Package**
   - Create a folder with your `manifest.json`
   - Add icon files (`color.png` 192x192, `outline.png` 32x32)
   - Zip the folder contents (not the folder itself)

2. **Upload to Teams**
   - Open Microsoft Teams
   - Go to "Apps" in the left sidebar
   - Click "Upload a custom app" → "Upload for [your org]"
   - Select your zip file
   - Click "Add" to install the bot

### Step 6: Local Development Setup

For local development, you'll need to expose your local server to the internet:

1. **Install ngrok** (or similar tunneling service)
   ```bash
   # Install ngrok
   npm install -g ngrok
   # or download from https://ngrok.com/
   ```

2. **Start your bot locally**
   ```bash
   python examples/teams/teams_agent.py
   ```

3. **Create ngrok tunnel**
   ```bash
   ngrok http 3978
   ```

4. **Update Azure Bot Configuration**
   - Copy the ngrok HTTPS URL (e.g., `https://abc123.ngrok.io`)
   - Update the messaging endpoint in Azure: `https://abc123.ngrok.io/api/messages`

## Environment Variables

Add the following to your `.env` in the project root:

- `MICROSOFT_APP_ID` — Application (client) ID from Azure App Registration
- `MICROSOFT_APP_PASSWORD` — Client secret from Azure App Registration
- `PORT` — Port for the web server (default: 3978)
- `JENTIC_AGENT_API_KEY` — Your Jentic Agent API key (get from app.jentic.com)
- `LLM_MODEL` — LLM model to use (optional)
- `TEAMS_DEV_MODE` — Set to "true" to skip authentication for development (NOT for production)

## Usage

### In Teams Channels or Group Chats
Mention the bot with your goal:
```
@StandardAgent find recent articles about machine learning
```

### Direct Messages
Send a direct message to the bot:
```
Help me plan a software architecture for a web application
```

### Commands
Use slash commands for configuration:
```
/help
/configure
/reasoner react
/reasoner list
/kill
```

## Deployment

### Deploy to Azure

1. **Create Azure Web App**
   ```bash
   az webapp create \
     --resource-group your-resource-group \
     --plan your-service-plan \
     --name your-app-name \
     --runtime "PYTHON|3.9"
   ```

2. **Configure Application Settings**
   ```bash
   az webapp config appsettings set \
     --resource-group your-resource-group \
     --name your-app-name \
     --settings \
       MICROSOFT_APP_ID="your-app-id" \
       MICROSOFT_APP_PASSWORD="your-app-password" \
       JENTIC_AGENT_API_KEY="your-api-key"
   ```

3. **Deploy Your Code**
   ```bash
   # Using Git deployment
   git remote add azure https://your-app-name.scm.azurewebsites.net:443/your-app-name.git
   git push azure main
   ```

4. **Update Bot Endpoint**
   - Update the messaging endpoint in Azure Bot to: `https://your-app-name.azurewebsites.net/api/messages`

### Deploy with Docker

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   COPY examples/teams/requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   
   EXPOSE 3978
   CMD ["python", "examples/teams/teams_agent.py"]
   ```

2. **Build and Run**
   ```bash
   docker build -t teams-agent .
   docker run -p 3978:3978 \
     -e MICROSOFT_APP_ID="your-app-id" \
     -e MICROSOFT_APP_PASSWORD="your-app-password" \
     -e JENTIC_AGENT_API_KEY="your-api-key" \
     teams-agent
   ```

## Troubleshooting

### Common Issues

1. **401 Unauthorized**
   - Check that `MICROSOFT_APP_ID` and `MICROSOFT_APP_PASSWORD` are correct
   - Verify the messaging endpoint URL is accessible
   - For development, ensure ngrok tunnel is active

2. **Bot not responding**
   - Check bot logs for errors
   - Verify the bot is running and accessible at the messaging endpoint
   - Test the health endpoint: `https://your-bot-url/health`

3. **Authentication errors**
   - Ensure app credentials are correctly configured
   - For development, you can set `TEAMS_DEV_MODE=true` to skip auth validation
   - Check that the Microsoft App ID in the manifest matches your Azure app

4. **Commands not working**
   - Verify the bot is properly installed in Teams
   - Check that command scope includes the context you're testing in
   - Ensure slash commands are properly formatted (no spaces after `/`)

### Debug Mode

For development, set these environment variables for additional logging:
```bash
export TEAMS_DEV_MODE=true
export LOG_LEVEL=DEBUG
```

### Health Check

The bot provides a health check endpoint at `/health` that returns:
```json
{
  "status": "healthy",
  "service": "teams-agent"
}
```

## Security Considerations

### Production Deployment

1. **Never skip authentication** in production (don't set `TEAMS_DEV_MODE=true`)
2. **Use secure credential storage** (Azure Key Vault, environment variables)
3. **Implement proper error handling** to avoid leaking sensitive information
4. **Use HTTPS** for all endpoints
5. **Validate and sanitize** all user inputs
6. **Monitor and log** bot activities for security auditing

### Rate Limiting

Consider implementing rate limiting to prevent abuse:
- Per-user rate limits
- Per-conversation rate limits
- Global rate limits

## Architecture

The Teams bot follows these patterns:

1. **Activity Handler**: Processes incoming Teams activities (messages, member additions, etc.)
2. **State Management**: Uses conversation and user state to track context
3. **Command Processing**: Handles slash commands for bot configuration
4. **Agent Integration**: Bridges Teams messages to the Standard Agent
5. **Error Handling**: Comprehensive error handling with proper logging
6. **Async Operations**: Non-blocking operations for better performance

## Contributing

When contributing to the Teams bot example:

1. Follow the existing code patterns from Slack/Discord examples
2. Ensure comprehensive error handling
3. Add proper logging for debugging
4. Update documentation for any new features
5. Test with actual Teams environment when possible

## Related Documentation

- [Microsoft Teams Bot Framework](https://docs.microsoft.com/en-us/microsoftteams/platform/bots/what-are-bots)
- [Azure Bot Service](https://docs.microsoft.com/en-us/azure/bot-service/)
- [Bot Framework SDK for Python](https://docs.microsoft.com/en-us/azure/bot-service/python/bot-builder-python-quickstart)
- [Standard Agent Documentation](../../README.md)
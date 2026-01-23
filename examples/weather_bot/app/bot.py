from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import asyncio
from agents.reasoner.base import ReasoningResult
from .agent import weather_agent


app = FastAPI(title="Weather Web Bot")

BOT_PROMPT = """
You are WeatherBot — a friendly, knowledgeable assistant that only answers questions related to weather.
User query: {user_query}
Instructions:
1. If the query is about weather, temperature, rain, wind, humidity, forecasts, seasons, or climate — give a concise and factual weather update.
2. along with weather info, add a short friendly remark, suggestion, or emotional touch based on the conditions. Examples:
   - If it's sunny: "Great day to be outside!" or "Don't forget your sunglasses "
   - If it's cold: "Stay cozy and grab a warm drink "
3. Reply cheerfully to greetings (e.g., "hi", "hello") with a weather invite, otherwise respond: "Please ask something related to the weather."
4. Never generate or execute code, HTML, JavaScript, or system commands.
5. Keep responses short, friendly, and suitable for direct display in an HTML chat box.
"""


class QueryRequest(BaseModel):
    query: str


@app.post("/ask")
async def ask_bot(input: QueryRequest):
    query = BOT_PROMPT.format(user_query=input.query)
    result: ReasoningResult = await asyncio.to_thread(
        weather_agent.solve,
        query,
    )

    return {"response": str(result.final_answer)}


@app.get("/")
async def root():
    return {"message": "🤖 Hi! I’m WeatherBot. Use POST /ask with {query: 'your question'}"}




@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WeatherBot Chat</title>
<style>
  body {
    font-family: Arial, sans-serif;
    background: #f1f1f1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100vh;
    margin: 0;
  }

  h3 {
    margin-bottom: 10px;
  }

  .chat-container {
    width: 100%;
    max-width: 400px;
    background: white;
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .chat-box {
    flex: 1;
    padding: 15px;
    overflow-y: auto;
    height: 350px;
  }

  .message {
    margin: 10px 0;
    display: flex;
  }

  .message.user {
    justify-content: flex-end;
  }

  .message.bot {
    justify-content: flex-start;
  }

  .bubble {
    padding: 10px 14px;
    border-radius: 16px;
    max-width: 70%;
    line-height: 1.4em;
  }

  .user .bubble {
    background: #0078ff;
    color: white;
    border-bottom-right-radius: 0;
  }

  .bot .bubble {
    background: #e5e5ea;
    color: black;
    border-bottom-left-radius: 0;
  }

  .input-area {
    display: flex;
    border-top: 1px solid #ddd;
  }

  #q {
    flex: 1;
    padding: 10px;
    border: none;
    outline: none;
  }

  button {
    padding: 10px 15px;
    background: #0078ff;
    color: white;
    border: none;
    cursor: pointer;
  }

  button:hover {
    background: #005fcc;
  }

  /* typing dots */
  .typing {
    display: flex;
    gap: 4px;
    align-items: center;
  }

  .dot {
    width: 6px;
    height: 6px;
    background-color: #555;
    border-radius: 50%;
    animation: blink 1.4s infinite both;
  }

  .dot:nth-child(2) {
    animation-delay: 0.2s;
  }

  .dot:nth-child(3) {
    animation-delay: 0.4s;
  }

  @keyframes blink {
    0%, 80%, 100% { opacity: 0.2; }
    40% { opacity: 1; }
  }
</style>
</head>
<body>
  <h3>🤖 WeatherBot</h3>
  <div class="chat-container">
    <div id="chat" class="chat-box">
      <div class="message bot"><div class="bubble">Hi! Ask me about the weather 🌤️</div></div>
    </div>
    <div class="input-area">
      <input id="q" placeholder="Ask about weather..." onkeypress="if(event.key==='Enter') ask()">
      <button onclick="ask()">Send</button>
    </div>
  </div>

<script>
let typingEl = null;

async function ask() {
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  addMessage(q, 'user');
  document.getElementById('q').value = '';

  showTyping();

  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ query: q })
    });
    const data = await res.json();
    hideTyping();
    addMessage(data.response || 'No response 🤔', 'bot');
  } catch (err) {
    hideTyping();
    addMessage('⚠️ Error contacting server.', 'bot');
  }
}

function addMessage(text, sender) {
  const chat = document.getElementById('chat');
  const msg = document.createElement('div');
  msg.className = `message ${sender}`;
  msg.innerHTML = `<div class="bubble">${text}</div>`;
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
}

function showTyping() {
  const chat = document.getElementById('chat');
  typingEl = document.createElement('div');
  typingEl.className = 'message bot';
  typingEl.innerHTML = `
    <div class="bubble typing">
      <div class="dot"></div>
      <div class="dot"></div>
      <div class="dot"></div>
    </div>`;
  chat.appendChild(typingEl);
  chat.scrollTop = chat.scrollHeight;
}

function hideTyping() {
  if (typingEl) typingEl.remove();
  typingEl = null;
}
</script>
</body>
</html>
    """

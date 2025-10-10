import os
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Load environment variables if .env exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

app = FastAPI(title="nullbyte Chatbot", version="0.3.0")

# Allow CORS (for frontend testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== Data Models ====
class ChatRequest(BaseModel):
    message: str
    user: Optional[str] = "guest"

class ChatResponse(BaseModel):
    reply: str
    user: str
    provider: str
    model: str

# ==== Local fallback (when no key is set) ====
def _local_demo_reply(message: str) -> str:
    text = message.strip()
    if not text:
        return "Say something and I'll respond."
    lower = text.lower()
    if any(x in lower for x in ["hello", "hi", "hey"]):
        return "Hello! I'm a local demo bot. Ask me anything."
    if lower.endswith("?"):
        return "Good question! I don’t have external knowledge here, but let’s reason it out together."
    if "help" in lower:
        return "You can ask me questions. Add your OpenRouter key to enable real AI."
    return f"You said: {text}"

# ==== Real AI via OpenRouter ====
def _openrouter_chat(message: str, system_prompt: Optional[str] = None) -> Optional[tuple[str, str]]:
  # Read and sanitize API key
  api_key_raw = os.getenv("OPENROUTER_API_KEY")
  api_key = api_key_raw.strip() if isinstance(api_key_raw, str) else None
  if not api_key:
    return None

  try:
    from openai import OpenAI

    base_url = "https://openrouter.ai/api/v1"
    model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-70b-instruct")
    system = system_prompt or os.getenv(
      "SYSTEM_PROMPT",
      "You are a helpful and friendly assistant created by Nullbyte.",
    )

    # Optional but recommended headers for OpenRouter attribution/rate-limits
    default_headers = {
      "HTTP-Referer": os.getenv("APP_URL", "http://localhost:8000"),
      "X-Title": os.getenv("APP_NAME", "nullbyte-chatbot"),
    }

    client = OpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers)

    resp = client.chat.completions.create(
      model=model,
      messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": message},
      ],
      temperature=0.7,
      max_tokens=400,
    )

    reply = (resp.choices[0].message.content or "").strip()
    return reply, model
  except Exception as e:
    print("Error using OpenRouter:", getattr(e, "message", str(e)))
    return None

# ==== Routes ====
@app.get("/health")
def health():
    provider = "openrouter" if os.getenv("OPENROUTER_API_KEY") else "local"
    model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-70b-instruct") if provider == "openrouter" else "demo"
    return {"status": "ok", "provider": provider, "model": model}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Try OpenRouter first
    oa = _openrouter_chat(req.message)
    if oa is not None:
        reply, model = oa
        return ChatResponse(
            reply=reply or _local_demo_reply(req.message),
            user=req.user or "guest",
            provider="openrouter",
            model=model,
        )

    # Fallback to demo mode
    reply = _local_demo_reply(req.message)
    return ChatResponse(reply=reply, user=req.user or "guest", provider="local", model="demo")

@app.get("/", response_class=HTMLResponse)
def ui_root():
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>nullbyte Chatbot</title>
    <style>
      :root { --bg:#0f172a;--panel:#111827;--text:#e5e7eb;--muted:#9ca3af;--accent:#22d3ee; }
      body { margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--text);}
      header {padding:16px;background:#0b1220;border-bottom:1px solid #1f2937;}
      header h1{margin:0;font-size:18px;font-weight:600;}
      .container{max-width:840px;margin:0 auto;padding:16px;}
      .chat{height:60vh;overflow-y:auto;padding:12px;background:var(--panel);border:1px solid #1f2937;border-radius:8px;}
      .msg{margin:8px 0;padding:10px 12px;border-radius:8px;max-width:80%;white-space:pre-wrap;}
      .user{background:#1f2937;margin-left:auto;}
      .bot{background:#0b1220;border:1px solid #1f2937;}
      form{display:flex;gap:8px;margin-top:12px;}
      input[type=text]{flex:1;padding:10px 12px;border-radius:8px;border:1px solid #1f2937;background:#0b1220;color:var(--text);}
      button{background:var(--accent);color:#0b1220;border:none;padding:10px 14px;border-radius:8px;font-weight:600;cursor:pointer;}
      button:disabled{opacity:.6;cursor:not-allowed;}
      .muted{color:var(--muted);font-size:12px;}
    </style>
  </head>
  <body>
    <header>
      <div class="container">
        <h1 id="title">nullbyte Chatbot</h1>
        <div id="subtitle" class="muted">Checking status…</div>
      </div>
    </header>
    <main class="container">
      <div id="chat" class="chat" aria-live="polite" aria-label="Chat messages"></div>
      <form id="form">
        <input id="input" type="text" placeholder="Type a message..." autocomplete="off" required />
        <button id="send" type="submit">Send</button>
      </form>
    </main>
    <script>
      const chat=document.getElementById('chat');
      const form=document.getElementById('form');
      const input=document.getElementById('input');
      const sendBtn=document.getElementById('send');
      const titleEl=document.getElementById('title');
      const subtitleEl=document.getElementById('subtitle');

      function addMsg(text,who){
        const div=document.createElement('div');
        div.className=`msg ${who}`;
        div.textContent=text;
        chat.appendChild(div);
        chat.scrollTop=chat.scrollHeight;
      }

      form.addEventListener('submit',async(e)=>{
        e.preventDefault();
        const text=input.value.trim();
        if(!text)return;
        addMsg(text,'user');
        input.value='';
        sendBtn.disabled=true;
        try{
          const res=await fetch('/chat',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({message:text,user:'guest'})
          });
          if(!res.ok)throw new Error('Request failed');
          const data=await res.json();
          addMsg(data.reply,'bot');
        }catch(err){
          addMsg('Error: '+(err?.message||err),'bot');
        }finally{
          sendBtn.disabled=false;
          input.focus();
        }
      });

      addMsg('Welcome! Ask me anything.','bot');
      fetch('/health').then(r=>r.json()).then(info=>{
        const provider=(info.provider||'local').toString();
        const model=(info.model||'demo').toString();
        document.title=`nullbyte Chatbot — AI: ${provider.toUpperCase()} (${model})`;
        titleEl.textContent=`nullbyte Chatbot — AI: ${provider.toUpperCase()} (${model})`;
        subtitleEl.textContent=provider==='local'
          ? 'No key required. Add an API key in .env to enable real AI.'
          : `Using ${provider.toUpperCase()} (${model})`;
      }).catch(()=>{subtitleEl.textContent='Status unavailable.';});
    </script>
  </body>
</html>
    """

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)

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

# Try to import Ollama (optional)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    ollama = None
    OLLAMA_AVAILABLE = False
    print("Ollama not available. Using alternative local AI...")

# Try to import transformers for local AI fallback
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
    print("Transformers available for local AI")
except ImportError:
    pipeline = None
    torch = None
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Install with: pip install transformers torch")

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

# ==== Local AI via Ollama ====
def _local_ollama_chat(message: str, system_prompt: Optional[str] = None) -> Optional[tuple[str, str]]:
    """Use Ollama for local AI inference"""
    if not OLLAMA_AVAILABLE or ollama is None:
        return None
        
    try:
        # Default model - you can change this to any model you have installed
        model = os.getenv("LOCAL_MODEL", "gemma3:1b")  # Your available model
        system = system_prompt or os.getenv(
            "SYSTEM_PROMPT",
            "You are a helpful and friendly assistant created by Nullbyte. You run completely locally without any API keys or internet connection required.",
        )
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": message}
        ]
        
        response = ollama.chat(
            model=model,
            messages=messages,
            options={
                "temperature": 0.7,
                "num_predict": 400,  # max tokens
            }
        )
        
        reply = response.get("message", {}).get("content", "").strip()
        return reply, model
        
    except Exception as e:
        print(f"Ollama error: {e}")
        # Check if it's a model not found error
        if "model" in str(e).lower() and "not found" in str(e).lower():
            print("Tip: Install a model with 'ollama pull llama3.2:3b'")
        return None

# ==== Alternative Local AI via Transformers ====
_cached_pipeline = None

def _local_transformers_chat(message: str, system_prompt: Optional[str] = None) -> Optional[tuple[str, str]]:
    """Use Transformers for local AI inference with error handling"""
    global _cached_pipeline
    
    if not TRANSFORMERS_AVAILABLE or pipeline is None or torch is None:
        return None
        
    try:
        # Initialize pipeline once and cache it
        if _cached_pipeline is None:
            model_name = os.getenv("HF_MODEL", "distilgpt2")
            print(f"Initializing model: {model_name}")
            
            # Try GPU first, fallback to CPU
            device = 0 if torch.cuda.is_available() else -1
            
            # Create the pipeline for this request
            _cached_pipeline = pipeline(
                "text-generation",
                model=model_name,
                device=device,
                pad_token_id=50256,
                truncation=True
            )
            print("Local AI model loaded and cached!")
        
        text_generator = _cached_pipeline
        
        # Format the prompt (keep it shorter for better performance)
        prompt = f"Human: {message}\nAI:"
        
        # Generate response with proper token settings
        outputs = text_generator(
            prompt,
            max_new_tokens=50,  # Use max_new_tokens instead of max_length
            num_return_sequences=1,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=50256,
            truncation=True,
            return_full_text=False  # Only return the generated part
        )
        
        # Extract and clean the response
        if outputs and len(outputs) > 0:
            response = outputs[0].get("generated_text", "").strip()
            
            # Clean up common artifacts
            response = response.split("Human:")[0].strip()
            response = response.split("AI:")[0].strip()
            response = response.split("\n")[0].strip()
            
            # Remove any remaining prompt artifacts
            if response.startswith("Human:") or response.startswith("AI:"):
                response = response.split(":", 1)[-1].strip()
            
            if not response or len(response) < 3:
                response = "I understand your message. Could you ask me something else?"
        else:
            response = "I'm having trouble generating a response right now."
        
        model_name = os.getenv("HF_MODEL", "distilgpt2")
        return response, model_name
        
    except Exception as e:
        print(f"Transformers error: {e}")
        return None

# ==== Simple fallback (when no AI is available) ====
def _simple_demo_reply(message: str) -> str:
    text = message.strip()
    if not text:
        return "Say something and I'll respond."
    lower = text.lower()
    if any(x in lower for x in ["hello", "hi", "hey"]):
        return "Hello! I'm running with basic responses. For better AI, install Ollama (https://ollama.com/download) or I can use Transformers for local AI."
    if lower.endswith("?"):
        return "Good question! I'm using simple responses right now. Install Ollama for more intelligent AI, or the Transformers library is working for local AI."
    if "help" in lower:
        return "Available options: 1) Install Ollama from https://ollama.com/download 2) Use Transformers (already installed) 3) Add OpenRouter API key for cloud AI."
    return f"You said: {text}. I'm using basic responses - install Ollama or use Transformers for smarter AI!"

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
    # Check if Ollama is available
    try:
        if OLLAMA_AVAILABLE and ollama is not None:
            ollama.list()
            provider = "ollama"
            model = os.getenv("LOCAL_MODEL", "gemma3:1b")
        else:
            raise Exception("Ollama not available")
    except Exception:
        # Check if Transformers is available
        if TRANSFORMERS_AVAILABLE:
            provider = "transformers"
            model = os.getenv("HF_MODEL", "distilgpt2")
        # Check if OpenRouter key exists as backup
        elif os.getenv("OPENROUTER_API_KEY"):
            provider = "openrouter"
            model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-70b-instruct")
        else:
            provider = "demo"
            model = "simple"
    
    return {"status": "ok", "provider": provider, "model": model}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Try Ollama first (local AI)
    ollama_result = _local_ollama_chat(req.message)
    if ollama_result is not None:
        reply, model = ollama_result
        return ChatResponse(
            reply=reply or _simple_demo_reply(req.message),
            user=req.user or "guest",
            provider="ollama",
            model=model,
        )
    
    # Try Transformers as second option (local AI)
    transformers_result = _local_transformers_chat(req.message)
    if transformers_result is not None:
        reply, model = transformers_result
        return ChatResponse(
            reply=reply or _simple_demo_reply(req.message),
            user=req.user or "guest",
            provider="transformers",
            model=model,
        )
    
    # Try OpenRouter as backup
    oa = _openrouter_chat(req.message)
    if oa is not None:
        reply, model = oa
        return ChatResponse(
            reply=reply or _simple_demo_reply(req.message),
            user=req.user or "guest",
            provider="openrouter",
            model=model,
        )

    # Fallback to simple demo mode
    reply = _simple_demo_reply(req.message)
    return ChatResponse(reply=reply, user=req.user or "guest", provider="demo", model="simple")

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
        const provider=(info.provider||'demo').toString();
        const model=(info.model||'simple').toString();
        document.title=`nullbyte Chatbot — AI: ${provider.toUpperCase()} (${model})`;
        titleEl.textContent=`nullbyte Chatbot — AI: ${provider.toUpperCase()} (${model})`;
        
        if(provider==='ollama'){
          subtitleEl.textContent=`Running locally with Ollama (${model}) - No API keys needed!`;
        }else if(provider==='transformers'){
          subtitleEl.textContent=`Running locally with Transformers (${model}) - No API keys needed!`;
        }else if(provider==='openrouter'){
          subtitleEl.textContent=`Using ${provider.toUpperCase()} (${model}) - Install local AI for offline use`;
        }else{
          subtitleEl.textContent='Install local AI: "pip install transformers torch" or get Ollama';
        }
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
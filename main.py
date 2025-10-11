import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import asyncio

# Load environment variables if .env exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Internet search capabilities
try:
    import requests
    from bs4 import BeautifulSoup
    import urllib.parse as urlparse
    INTERNET_AVAILABLE = True
    print("Internet search available")
except ImportError as e:
    requests = None
    BeautifulSoup = None
    urlparse = None
    INTERNET_AVAILABLE = False
    print(f"Internet search not available: {e}")

# Import NKOD integration
try:
    from nkod_integration import get_nkod_manager, NKODDataManager
    NKOD_AVAILABLE = True
    print("NKOD integration available")
except ImportError as e:
    get_nkod_manager = None
    NKOD_AVAILABLE = False
    print(f"NKOD integration not available: {e}")

# Import Úřední Desky integration
try:
    from uredni_desky import get_uredni_desky_manager, UredniDeskyManager
    UREDNI_DESKY_AVAILABLE = True
    print("Uredni Desky integration available")
except ImportError as e:
    get_uredni_desky_manager = None
    UREDNI_DESKY_AVAILABLE = False
    print(f"Uredni Desky integration not available: {e}")

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
    # Temporarily disable transformers due to import issues
    # from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    # import torch
    # TRANSFORMERS_AVAILABLE = True
    # print("Transformers available for local AI")
    raise ImportError("Transformers temporarily disabled")
except ImportError:
    pipeline = None
    torch = None
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Install with: pip install transformers torch")

app = FastAPI(
    title="nullbyte Chatbot - Český AI Asistent", 
    version="0.4.0",
    description="Český AI chatbot s integrací NKOD (Národní katalog otevřených dat)"
)

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
    datasets: Optional[List[dict]] = None  # For NKOD search results

class NKODSearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5

class UredniDeskyRequest(BaseModel):
    municipality: Optional[str] = None
    municipalities: Optional[List[str]] = None
    max_documents: Optional[int] = 50

class UredniDeskyResponse(BaseModel):
    success: bool
    message: str
    data: Dict[str, Any]

class UredniDeskySearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5

# ==== Local AI via Ollama ====
def _local_ollama_chat(message: str, system_prompt: Optional[str] = None) -> Optional[tuple[str, str]]:
    """Use Ollama for local AI inference"""
    if not OLLAMA_AVAILABLE or ollama is None:
        return None
        
    try:
        # Default model - you can change this to any model you have installed
        model = os.getenv("LOCAL_MODEL", "gemma3:4b")  # Your available model
        system = system_prompt or os.getenv(
            "SYSTEM_PROMPT",
            "Jsi přesný AI asistent s přístupem k internetu a českým datům. PRAVIDLA: 1) Pouze česká odpověď 2) Buď přímý a konkrétní 3) Používej aktuální data z internetu 4) Cituj zdroje 5) Pokud nevíš, řekni to přímo 6) Strukturované odpovědi s číselnými údaji 7) Maximální přesnost. Odpovídej jasně a věcně.",
        )
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": message}
        ]
        
        response = ollama.chat(
            model=model,
            messages=messages,
            options={
                "temperature": 0.3,  # Lower temperature for more accurate, focused responses
                "num_predict": 2000,  # Higher token limit for complete responses
                "top_k": 20,  # More focused vocabulary selection
                "top_p": 0.8,  # More deterministic responses
                "repeat_penalty": 1.15,  # Stronger penalty against repetition
                "presence_penalty": 0.1,  # Encourage diverse topics
                "frequency_penalty": 0.1,  # Reduce repetitive phrases
                "stop": [],  # Don't stop early
                "seed": 42,  # Consistent results for similar queries
            }
        )
        
        reply = response.get("message", {}).get("content", "").strip()
        
        # Validate and improve response quality
        reply = _validate_and_improve_response(reply, message)
        
        return reply, model
        
    except Exception as e:
        print(f"Ollama error: {e}")
        # Check if it's a model not found error
        if "model" in str(e).lower() and "not found" in str(e).lower():
            print("Tip: Install a model with 'ollama pull llama3.2:3b'")
        return None

# ==== Internet Search Functions (DISABLED - NKOD-only mode) ====
def _search_internet(query: str, num_results: int = 3) -> List[dict]:
  """Internet search is disabled in NKOD-only mode. Return empty list."""
  return []


def _get_current_czech_stats(topic: str) -> dict:
  """Stub for current Czech stats: in NKOD-only mode we return an empty dict.

  This preserves the function signature for compatibility but prevents
  any external network access.
  """
  return {}

# ==== Response Validation and Improvement ====
def _validate_and_improve_response(reply: str, original_query: str) -> str:
    """Validate and improve AI response quality"""
    if not reply or len(reply.strip()) < 10:
        return "Omlouvám se, ale nemohu vygenerovat odpovídající odpověď. Můžete prosím přeformulovat svou otázku?"
    
    # Check for Czech language
    czech_indicators = ['a', 'je', 'na', 'v', 'z', 'do', 'od', 'po', 'při', 'pro', 'se', 'si', 'že', 'nebo', 'ale']
    words = reply.lower().split()[:20]  # Check first 20 words
    czech_count = sum(1 for word in words if word in czech_indicators)
    
    if len(words) > 5 and czech_count < 2:
        reply = "Odpovídám v češtině: " + reply
    
    # Ensure proper structure for data queries
    if any(keyword in original_query.lower() for keyword in ['kolik', 'počet', 'statistik', 'data', 'údaje']):
        if not any(marker in reply for marker in ['•', '-', '1.', '2.', 'Podle']):
            reply = "**Strukturovaná odpověď:**\n\n" + reply
    
    # Add uncertainty markers if response seems uncertain
    uncertainty_markers = ['možná', 'pravděpodobně', 'asi', 'nejspíš']
    if any(marker in reply.lower() for marker in uncertainty_markers):
        if "nejsem si jistý" not in reply.lower():
            reply += "\n\n⚠️ *Upozornění: Některé informace mohou vyžadovat ověření.*"
    
    return reply.strip()

# ==== NKOD Enhanced AI Chat ====
def _enhance_with_nkod_data(message: str, ai_response: str) -> tuple[str, List[dict]]:
  """
  Build an NKOD-only enhanced reply. This intentionally ignores internet and
  úřední desky results to guarantee NKOD-only RAG operation.
  Returns enhanced_response and the list of NKOD datasets used.
  """
  datasets: List[dict] = []

  try:
    # 1) Fetch NKOD datasets (RAG) if available
    if NKOD_AVAILABLE and get_nkod_manager is not None:
      nkod_manager = get_nkod_manager()
      datasets = nkod_manager.search_datasets(message, n_results=5)

    # 2) Compute NKOD-only accuracy
    accuracy_score = _calculate_simple_accuracy(message, ai_response, datasets, [], [])

    # 3) Build enhanced reply
    enhanced_response = ai_response or ""
    enhanced_response += f"\n\n**🎯 PŘESNOST: {accuracy_score}%**\n"

    # 4) Add NKOD datasets summary
    if datasets:
      enhanced_response += "\n📊 **Oficiální datasety (NKOD):**\n"
      for i, dataset in enumerate(datasets[:3], 1):
        title = dataset.get('title', 'Bez názvu')
        publisher = dataset.get('publisher', 'Neznámý vydavatel')
        similarity = int(float(dataset.get('similarity', 0.0)) * 100)
        enhanced_response += f"{i}. **{title}** — {publisher} ({similarity}% relevance)\n"

    # 5) Source attribution (NKOD-only)
    sources = ["NKOD (oficiální datasety)", "NKOD-RAG AI"]
    enhanced_response += f"\n**📚 Zdroje:** {' + '.join(sources)}\n"

    return enhanced_response, datasets

  except Exception as e:
    print(f"Error enhancing response (NKOD-only): {e}")
    # Fallback: return basic ai_response and empty datasets
    fallback = ai_response or "Nemám dost informací z NKOD."
    fallback += "\n\n**🎯 PŘESNOST: 70%**\n**📚 Zdroje:** NKOD (pokud dostupné)"
    return fallback, []

def _calculate_simple_accuracy(message: str, ai_response: str, datasets: List[dict], web_results: List[dict], uredni_desky_results: Optional[List[dict]] = None) -> int:
    """Calculate straightforward accuracy percentage (NKOD-only).

    This function intentionally ignores internet and úřední desky results and
    computes an accuracy score solely based on NKOD dataset relevance and
    response structure.
    """
    base_accuracy = 75

    # Boost for NKOD data only
    if datasets:
        try:
            avg_similarity = sum(float(d.get('similarity', 0.0)) for d in datasets) / max(1, len(datasets))
        except Exception:
            avg_similarity = 0.0
        nkod_boost = int(avg_similarity * 15)  # NKOD relevance can add up to ~15%
        base_accuracy += nkod_boost

    # Response quality check (structure helps)
    if ai_response and len(ai_response) > 120 and any(marker in ai_response for marker in ['podle', '•', '**', '1.']):
        base_accuracy += 3

    # Keep accuracy in a sensible NKOD-only range
    return max(60, min(95, base_accuracy))

# Removed complex accuracy functions - now using simpler _calculate_simple_accuracy

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
        prompt = f"Člověk: {message}\nAsistent:"
        
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
            response = response.split("Člověk:")[0].strip()
            response = response.split("Asistent:")[0].strip()
            response = response.split("\n")[0].strip()
            
            # Remove any remaining prompt artifacts
            if response.startswith("Člověk:") or response.startswith("Asistent:"):
                response = response.split(":", 1)[-1].strip()
            
            if not response or len(response) < 3:
                response = "Rozumím vaší zprávě. Můžete se mě zeptat na něco jiného?"
        else:
            response = "Momentálně mám problém s generováním odpovědi."
        
        model_name = os.getenv("HF_MODEL", "distilgpt2")
        return response, model_name
        
    except Exception as e:
        print(f"Transformers error: {e}")
        return None

# ==== Jednoduchá záložní odpověď (když není k dispozici AI) ====
def _simple_demo_reply(message: str) -> str:
    text = message.strip()
    if not text:
        return "Řekněte něco a já odpovím."
    lower = text.lower()
    if any(x in lower for x in ["ahoj", "zdravím", "dobrý den", "čau", "hello", "hi", "hey"]):
        return "Ahoj! Běžím se základními odpověďmi. Pro lepší AI nainstalujte Ollama (https://ollama.com/download) nebo mohu použít Transformers pro lokální AI."
    if lower.endswith("?"):
        return "Dobrá otázka! Momentálně používám jednoduché odpovědi. Nainstalujte Ollama pro inteligentnější AI, nebo knihovna Transformers funguje pro lokální AI."
    if any(x in lower for x in ["pomoc", "help", "nápověda"]):
        return "Dostupné možnosti: 1) Nainstalovat Ollama z https://ollama.com/download 2) Použít Transformers (už nainstalováno) 3) Přidat OpenRouter API klíč pro cloudové AI."
    return f"Řekli jste: {text}. Používám základní odpovědi - nainstalujte Ollama nebo použijte Transformers pro chytřejší AI!"

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
      "Jsi přesný AI s přístupem k internetu a českým datům. Odpovídáš pouze v češtině, přímo a věcně. Používáš aktuální data z internetu. Cituj zdroje. Pokud nevíš, řekni to. Maximální přesnost.",
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
      max_tokens=2000,  # Much higher limit for complete responses
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
            model = os.getenv("LOCAL_MODEL", "gemma3:4b")
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

@app.post("/nkod/search")
def search_nkod(req: NKODSearchRequest):
    """Vyhledávání NKOD datasetů"""
    if not NKOD_AVAILABLE or get_nkod_manager is None:
        return JSONResponse(
            status_code=503,
            content={"error": "NKOD integrace není dostupná"}
        )
    
    try:
        nkod_manager = get_nkod_manager()
        datasets = nkod_manager.search_datasets(req.query, req.limit or 5)
        return {"datasets": datasets, "query": req.query, "count": len(datasets)}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Vyhledávání selhalo: {str(e)}"}
        )

@app.post("/nkod/refresh")
def refresh_nkod_data(background_tasks: BackgroundTasks):
    """Obnovení NKOD dat na pozadí"""
    if not NKOD_AVAILABLE or get_nkod_manager is None:
        return JSONResponse(
            status_code=503,
            content={"error": "NKOD integrace není dostupná"}
        )
    
    async def refresh_task():
        try:
            if get_nkod_manager is not None:
                nkod_manager = get_nkod_manager()
                success = await nkod_manager.refresh_data()
                return success
            return False
        except Exception as e:
            print(f"Background refresh failed: {e}")
            return False
    
    background_tasks.add_task(refresh_task)
    return {"message": "Obnovování dat bylo spuštěno na pozadí"}

@app.get("/nkod/stats")
def nkod_stats():
    """Získání statistik NKOD integrace"""
    if not NKOD_AVAILABLE or get_nkod_manager is None:
        return {"available": False, "error": "NKOD integrace není dostupná"}
    
    try:
        nkod_manager = get_nkod_manager()
        stats = nkod_manager.get_stats()
        stats["available"] = True
        return stats
    except Exception as e:
        return {"available": False, "error": str(e)}

# ==== Úřední Desky Endpoints ====
@app.post("/uredni-desky/collect", response_model=UredniDeskyResponse)
async def collect_uredni_desky(req: UredniDeskyRequest) -> UredniDeskyResponse:
    """Collect úřední desky documents for specified municipality/municipalities"""
    try:
        if not UREDNI_DESKY_AVAILABLE or get_uredni_desky_manager is None:
            return UredniDeskyResponse(
                success=False,
                message="Úřední desky manager není k dispozici",
                data={"documents_processed": 0, "municipalities": []}
            )
        
        uredni_manager = get_uredni_desky_manager()
        
        # Process single municipality or multiple
        municipalities = req.municipalities if req.municipalities else [req.municipality] if req.municipality else []
        
        if not municipalities:
            return UredniDeskyResponse(
                success=False,
                message="Není specifikována žádná obec/město",
                data={"documents_processed": 0, "municipalities": []}
            )
        
        total_processed = 0
        processed_municipalities = []
        
        for municipality in municipalities:
            try:
                result = await uredni_manager.collect_municipality_data(municipality)
                count = result.get('documents_processed', 0)
                total_processed += count
                processed_municipalities.append({
                    "name": municipality,
                    "documents": count,
                    "success": True
                })
            except Exception as e:
                processed_municipalities.append({
                    "name": municipality,
                    "documents": 0,
                    "success": False,
                    "error": str(e)
                })
        
        return UredniDeskyResponse(
            success=True,
            message=f"Úspěšně zpracováno {total_processed} dokumentů z {len([m for m in processed_municipalities if m['success']])} obcí",
            data={
                "documents_processed": total_processed,
                "municipalities": processed_municipalities
            }
        )
        
    except Exception as e:
        return UredniDeskyResponse(
            success=False,
            message=f"Chyba při sběru dat: {str(e)}",
            data={"documents_processed": 0, "municipalities": []}
        )

@app.get("/uredni-desky/search", response_model=UredniDeskyResponse)
async def search_uredni_desky(query: str, n_results: int = 5, municipality: Optional[str] = None) -> UredniDeskyResponse:
    """Search úřední desky documents using RAG"""
    try:
        if not UREDNI_DESKY_AVAILABLE or get_uredni_desky_manager is None:
            return UredniDeskyResponse(
                success=False,
                message="Úřední desky manager není k dispozici",
                data={"results": []}
            )
        
        uredni_manager = get_uredni_desky_manager()
        results = uredni_manager.search_rag_documents(query, n_results=n_results)
        
        # Format results for response
        formatted_results = []
        for result in results:
            formatted_results.append({
                "content": result['content'][:300] + "..." if len(result['content']) > 300 else result['content'],
                "similarity": round(result['similarity'], 3),
                "metadata": result['metadata']
            })
        
        return UredniDeskyResponse(
            success=True,
            message=f"Nalezeno {len(results)} relevantních dokumentů",
            data={"results": formatted_results}
        )
        
    except Exception as e:
        return UredniDeskyResponse(
            success=False,
            message=f"Chyba při vyhledávání: {str(e)}",
            data={"results": []}
        )

@app.get("/uredni-desky/stats", response_model=UredniDeskyResponse)  
async def get_uredni_desky_stats() -> UredniDeskyResponse:
    """Get statistics about collected úřední desky documents"""
    try:
        if not UREDNI_DESKY_AVAILABLE or get_uredni_desky_manager is None:
            return UredniDeskyResponse(
                success=False,
                message="Úřední desky manager není k dispozici",
                data={"stats": {}}
            )
        
        uredni_manager = get_uredni_desky_manager()
        stats = uredni_manager.get_rag_stats()
        
        return UredniDeskyResponse(
            success=True,
            message="Statistiky úspěšně načteny",
            data={"stats": stats}
        )
        
    except Exception as e:
        return UredniDeskyResponse(
            success=False,
            message=f"Chyba při načítání statistik: {str(e)}",
            data={"stats": {}}
        )

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
  """
  NKOD-only chat endpoint.

  This route deliberately ignores any internet searches or alternative AI
  providers. It will generate a reply using the following rules:
    1. If NKOD integration is available, use NKOD datasets to build a
     structured response (RAG) and compute accuracy based on NKOD only.
    2. If NKOD is not available, fall back to a simple canned reply but
     still attempt to attach any NKOD datasets if present.
  """
  # Prefer NKOD manager for RAG-enhanced replies
  datasets: List[dict] = []
  nkod_reply = None

  if NKOD_AVAILABLE and get_nkod_manager is not None:
    try:
      nkod_manager = get_nkod_manager()
      # Use NKOD RAG to retrieve relevant datasets and short summary
      datasets = nkod_manager.search_datasets(req.message, n_results=5)

      # Build a deterministic NKOD-based reply if NKOD provides summaries
      if datasets:
        top = datasets[0]
        title = top.get('title', 'bez názvu')
        publisher = top.get('publisher', 'Neznámý vydavatel')
        nkod_reply = f"Našel jsem oficiální dataset: {title} (vydavatel: {publisher}). Mohu načíst více informací nebo poskytnout odkaz na dataset."
    except Exception as e:
      print(f"NKOD retrieval error: {e}")

  # If no NKOD reply produced, use simple demo reply (offline safe)
  if not nkod_reply:
    nkod_reply = _simple_demo_reply(req.message)

  # Enhance response using NKOD-only enhancer (this will ignore internet/úřední desky)
  enhanced_reply, used_datasets = _enhance_with_nkod_data(req.message, nkod_reply)

  return ChatResponse(
    reply=enhanced_reply,
    user=req.user or "guest",
    provider="nkod-rag",
    model="nkod-v1",
    datasets=used_datasets
  )

@app.get("/", response_class=HTMLResponse)
def ui_root():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>nullbyte Chatbot - Český AI Asistent</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <!-- Load Markdown parser and sanitizer -->
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/dompurify@3.0.5/dist/purify.min.js"></script>
  <style>
    :root {
      --bg: #0f172a;
      --panel: rgba(17, 24, 39, 0.7);
      --text: #e5e7eb;
      --muted: #9ca3af;
      --accent: #22d3ee;
      --accent-dark: #06b6d4;
      --user-bg: rgba(31, 41, 55, 0.8);
      --bot-bg: rgba(11, 18, 32, 0.8);
      --border: rgba(31, 41, 55, 0.5);
      --success: #10b981;
      --warning: #f59e0b;
      --code-bg: rgba(15, 23, 42, 0.8);
    }
    
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }
    
    body {
      margin: 0;
      font-family: 'Segoe UI', system-ui, -apple-system, Roboto, sans-serif;
      background: linear-gradient(135deg, #0c1120 0%, #1a2238 100%);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
      line-height: 1.6;
    }
    
    .background-elements {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      z-index: -1;
      overflow: hidden;
    }
    
    .bg-circle {
      position: absolute;
      border-radius: 50%;
      background: radial-gradient(circle, var(--accent) 0%, transparent 70%);
      opacity: 0.1;
    }
    
    .bg-circle:nth-child(1) {
      width: 300px;
      height: 300px;
      top: 10%;
      left: 5%;
    }
    
    .bg-circle:nth-child(2) {
      width: 200px;
      height: 200px;
      bottom: 15%;
      right: 10%;
    }
    
    .bg-circle:nth-child(3) {
      width: 150px;
      height: 150px;
      top: 40%;
      right: 20%;
    }
    
    header {
      padding: 20px 24px;
      background: rgba(11, 18, 32, 0.7);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid var(--border);
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
      position: sticky;
      top: 0;
      z-index: 100;
    }
    
    .header-content {
      max-width: 840px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    
    header h1 {
      margin: 0;
      font-size: 22px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 12px;
    }
    
    header h1 i {
      color: var(--accent);
    }
    
    #subtitle {
      color: var(--muted);
      font-size: 14px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    
    .container {
      max-width: 840px;
      margin: 0 auto;
      padding: 24px;
      flex: 1;
      display: flex;
      flex-direction: column;
    }
    
    .chat {
      height: 60vh;
      overflow-y: auto;
      padding: 16px;
      background: var(--panel);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border);
      border-radius: 16px;
      margin-bottom: 24px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    
    .msg {
      padding: 16px;
      border-radius: 18px;
      max-width: 85%;
      position: relative;
      animation: fadeIn 0.3s ease-out;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
      display: flex;
      gap: 16px;
    }
    
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
    
    .avatar {
      width: 40px;
      height: 40px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }
    
    .user .avatar {
      background: linear-gradient(135deg, #3b82f6, #1d4ed8);
    }
    
    .bot .avatar {
      background: linear-gradient(135deg, var(--accent), var(--accent-dark));
    }
    
    .msg-content {
      flex: 1;
      overflow-wrap: break-word;
    }
    
    /* Markdown styling */
    .msg-content h1,
    .msg-content h2,
    .msg-content h3,
    .msg-content h4,
    .msg-content h5,
    .msg-content h6 {
      margin: 16px 0 12px;
      color: var(--text);
    }
    
    .msg-content p {
      margin: 8px 0;
    }
    
    .msg-content ul,
    .msg-content ol {
      padding-left: 24px;
      margin: 12px 0;
    }
    
    .msg-content li {
      margin: 6px 0;
    }
    
    .msg-content a {
      color: var(--accent);
      text-decoration: underline;
    }
    
    .msg-content strong {
      font-weight: 700;
    }
    
    .msg-content em {
      font-style: italic;
    }
    
    .msg-content blockquote {
      border-left: 3px solid var(--accent);
      padding-left: 16px;
      margin: 12px 0;
      color: var(--muted);
    }
    
    .msg-content code {
      background: var(--code-bg);
      padding: 2px 6px;
      border-radius: 4px;
      font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
      font-size: 0.95em;
    }
    
    .msg-content pre {
      background: var(--code-bg);
      padding: 16px;
      border-radius: 10px;
      overflow-x: auto;
      margin: 16px 0;
    }
    
    .msg-content pre code {
      background: none;
      padding: 0;
      font-size: 0.95em;
    }
    
    .user {
      background: var(--user-bg);
      margin-left: auto;
      border-bottom-right-radius: 6px;
    }
    
    .bot {
      background: var(--bot-bg);
      border-bottom-left-radius: 6px;
    }
    
    form {
      display: flex;
      gap: 12px;
      margin-top: auto;
    }
    
    input[type="text"] {
      flex: 1;
      padding: 14px 20px;
      border-radius: 50px;
      border: 1px solid var(--border);
      background: rgba(11, 18, 32, 0.7);
      color: var(--text);
      font-size: 16px;
      transition: all 0.3s ease;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    input[type="text"]:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.3);
    }
    
    button {
      background: linear-gradient(135deg, var(--accent), var(--accent-dark));
      color: #0b1220;
      border: none;
      padding: 14px 28px;
      border-radius: 50px;
      font-weight: 700;
      cursor: pointer;
      font-size: 16px;
      transition: all 0.3s ease;
      box-shadow: 0 4px 15px rgba(34, 211, 238, 0.3);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    
    button:hover:not(:disabled) {
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(34, 211, 238, 0.4);
    }
    
    button:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
      box-shadow: none;
    }
    
    .muted {
      color: var(--muted);
      font-size: 14px;
    }
    
    .datasets {
      margin-top: 24px;
      padding: 20px;
      background: var(--panel);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border);
      border-radius: 16px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
      animation: slideUp 0.4s ease-out;
    }
    
    @keyframes slideUp {
      from { opacity: 0; transform: translateY(20px); }
      to { opacity: 1; transform: translateY(0); }
    }
    
    .datasets h3 {
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 20px;
    }
    
    .datasets-list {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 16px;
    }
    
    .dataset {
      padding: 18px;
      background: rgba(31, 41, 55, 0.6);
      border-radius: 14px;
      transition: transform 0.3s ease, box-shadow 0.3s ease;
      border: 1px solid var(--border);
    }
    
    .dataset:hover {
      transform: translateY(-4px);
      box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
      border-color: var(--accent);
    }
    
    .dataset strong {
      display: block;
      margin-bottom: 8px;
      font-size: 16px;
      color: var(--text);
    }
    
    .dataset small {
      display: block;
      margin-bottom: 12px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }
    
    .dataset em {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--accent);
      font-style: normal;
      font-size: 13px;
    }
    
    .nkod-controls {
      display: flex;
      gap: 12px;
      margin-top: 24px;
      flex-wrap: wrap;
    }
    
    .nkod-btn {
      background: rgba(55, 65, 81, 0.7);
      color: var(--text);
      border: 1px solid var(--border);
      padding: 12px 20px;
      font-size: 14px;
      border-radius: 12px;
      cursor: pointer;
      transition: all 0.3s ease;
      display: flex;
      align-items: center;
      gap: 8px;
      flex: 1;
      min-width: 180px;
    }
    
    .nkod-btn:hover {
      background: rgba(75, 85, 99, 0.8);
      border-color: var(--accent);
      transform: translateY(-2px);
    }
    
    .status-indicator {
      display: inline-block;
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--success);
      margin-right: 6px;
    }
    
    .error-indicator {
      background: #ef4444;
    }
    
    .warning-indicator {
      background: var(--warning);
    }
    
    ::-webkit-scrollbar {
      width: 8px;
    }
    
    ::-webkit-scrollbar-track {
      background: rgba(0, 0, 0, 0.1);
      border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
      background: rgba(34, 211, 238, 0.5);
      border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
      background: var(--accent);
    }
    
    @media (max-width: 768px) {
      .container {
        padding: 16px;
      }
      
      .chat {
        height: 55vh;
        padding: 12px;
      }
      
      .msg {
        max-width: 90%;
        padding: 14px;
      }
      
      .datasets-list {
        grid-template-columns: 1fr;
      }
      
      form {
        flex-direction: column;
      }
      
      button {
        padding: 16px;
        justify-content: center;
      }
      
      .nkod-controls {
        flex-direction: column;
      }
      
      .nkod-btn {
        min-width: auto;
      }
    }
  </style>
</head>
<body>
  <div class="background-elements">
    <div class="bg-circle"></div>
    <div class="bg-circle"></div>
    <div class="bg-circle"></div>
  </div>
  
  <header>
    <div class="header-content">
      <h1><i class="fas fa-robot"></i> nullbyte Chatbot - Český AI Asistent</h1>
      <div id="subtitle" class="muted"><span class="status-indicator"></span>Kontroluje se stav…</div>
    </div>
  </header>
  
  <main class="container">
    <div id="chat" class="chat" aria-live="polite" aria-label="Chat messages"></div>
    
    <div id="datasets" class="datasets" style="display:none;">
      <h3><i class="fas fa-database"></i> 📊 Související datasety z data.gov.cz</h3>
      <div id="datasets-list" class="datasets-list"></div>
    </div>
    
    <form id="form">
      <input id="input" type="text" placeholder="Zeptejte se na otevřená data, statistiky nebo cokoliv jiného..." autocomplete="off" required />
      <button id="send" type="submit"><i class="fas fa-paper-plane"></i> Odeslat</button>
    </form>
    
    <div class="nkod-controls">
      <button id="refresh-nkod" class="nkod-btn"><i class="fas fa-sync-alt"></i> 🔄 Obnovit NKOD data</button>
      <button id="nkod-stats" class="nkod-btn"><i class="fas fa-chart-line"></i> 📈 NKOD statistiky</button>
    </div>
  </main>
  
  <script>
    // Configure marked for safer rendering
    marked.setOptions({
      breaks: true,
      gfm: true,
      sanitize: false // We'll use DOMPurify instead
    });

    const chat = document.getElementById('chat');
    const form = document.getElementById('form');
    const input = document.getElementById('input');
    const sendBtn = document.getElementById('send');
    const titleEl = document.querySelector('header h1');
    const subtitleEl = document.getElementById('subtitle');
    const datasetsEl = document.getElementById('datasets');
    const datasetsListEl = document.getElementById('datasets-list');
    const refreshBtn = document.getElementById('refresh-nkod');
    const statsBtn = document.getElementById('nkod-stats');
    const statusIndicator = document.querySelector('.status-indicator');

    function addMsg(text, who, datasets = null) {
      const div = document.createElement('div');
      div.className = `msg ${who}`;
      
      const avatar = document.createElement('div');
      avatar.className = 'avatar';
      avatar.innerHTML = who === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
      
      const content = document.createElement('div');
      content.className = 'msg-content';
      
      if (who === 'bot') {
        // Render Markdown for bot responses
        const rawHtml = marked.parse(text);
        const cleanHtml = DOMPurify.sanitize(rawHtml, {
          ALLOWED_TAGS: ['p', 'strong', 'em', 'ul', 'ol', 'li', 'br', 'a', 'code', 'pre', 'blockquote', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'],
          ALLOWED_ATTR: ['href', 'target']
        });
        content.innerHTML = cleanHtml;
      } else {
        // User messages are plain text (no HTML)
        content.textContent = text;
      }
      
      div.appendChild(avatar);
      div.appendChild(content);
      chat.appendChild(div);
      
      if (datasets && datasets.length > 0) {
        showDatasets(datasets);
      } else {
        datasetsEl.style.display = 'none';
      }
      
      chat.scrollTop = chat.scrollHeight;
    }

    function showDatasets(datasets) {
      datasetsListEl.innerHTML = '';
      datasets.forEach((dataset, i) => {
        const div = document.createElement('div');
        div.className = 'dataset';
        div.innerHTML = `
          <strong>${dataset.title || 'Bez názvu'}</strong><br>
          <small>${dataset.description || 'Bez popisu'}</small><br>
          <em><i class="fas fa-building"></i> ${dataset.publisher || 'Neznámý vydavatel'}</em>
        `;
        datasetsListEl.appendChild(div);
      });
      datasetsEl.style.display = 'block';
    }

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const text = input.value.trim();
      if (!text) return;
      addMsg(text, 'user');
      input.value = '';
      sendBtn.disabled = true;
      try {
        const res = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, user: 'guest' })
        });
        if (!res.ok) throw new Error('Request failed');
        const data = await res.json();
        addMsg(data.reply, 'bot', data.datasets);
      } catch (err) {
        addMsg('Chyba: ' + (err?.message || err), 'bot');
      } finally {
        sendBtn.disabled = false;
        input.focus();
      }
    });

    refreshBtn.addEventListener('click', async () => {
      refreshBtn.disabled = true;
      refreshBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Obnovování...';
      try {
        const res = await fetch('/nkod/refresh', { method: 'POST' });
        const data = await res.json();
        addMsg(data.message || 'Obnovování dat bylo spuštěno', 'bot');
      } catch (err) {
        addMsg('Chyba při obnovování NKOD dat: ' + err.message, 'bot');
      } finally {
        refreshBtn.disabled = false;
        refreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i> 🔄 Obnovit NKOD data';
      }
    });

    statsBtn.addEventListener('click', async () => {
      try {
        const res = await fetch('/nkod/stats');
        const stats = await res.json();
        if (stats.available) {
          addMsg(`📈 NKOD statistiky:\n• Indexované datasety: ${stats.indexed_datasets || 0}\n• Cache: ${stats.cache_info?.cached_count || 0} datasetů\n• Poslední aktualizace: ${stats.cache_info?.last_updated || 'Nikdy'}`, 'bot');
        } else {
          addMsg('NKOD integrace není dostupná: ' + stats.error, 'bot');
        }
      } catch (err) {
        addMsg('Chyba při získávání NKOD statistik: ' + err.message, 'bot');
      }
    });

    addMsg('Vítejte! Zeptejte se mě na cokoliv. Mohu vám pomoci s českými otevřenými daty, statistikami nebo běžnými otázkami.', 'bot');
    
    fetch('/health')
      .then(r => r.json())
      .then(info => {
        const provider = (info.provider || 'demo').toString();
        const model = (info.model || 'simple').toString();
        document.title = `nullbyte Chatbot — AI: ${provider.toUpperCase()} (${model})`;
        titleEl.innerHTML = `<i class="fas fa-robot"></i> nullbyte Chatbot — AI: ${provider.toUpperCase()} (${model})`;
        
        if (provider === 'ollama') {
          subtitleEl.innerHTML = `<span class="status-indicator"></span>Běžím lokálně na Ollama (${model}).`;
        } else if (provider === 'transformers') {
          subtitleEl.innerHTML = `<span class="status-indicator"></span>Běžím lokálně s Transformers (${model}).`;
        } else if (provider === 'openrouter') {
          subtitleEl.innerHTML = `<span class="status-indicator warning-indicator"></span>Používám ${provider.toUpperCase()} (${model}) - Pro offline použití nainstalujte lokální AI`;
        } else {
          subtitleEl.innerHTML = `<span class="status-indicator error-indicator"></span>Nainstalujte lokální AI: "pip install transformers torch" nebo získejte Ollama`;
        }
      })
      .catch(() => {
        subtitleEl.innerHTML = `<span class="status-indicator error-indicator"></span>Stav není dostupný.`;
      });
  </script>
</body>
</html>
    """

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        timeout_keep_alive=120,  # Keep connections alive longer
    )
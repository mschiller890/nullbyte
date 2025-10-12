"""
Úřední deska Chatbot - Kompletní řešení s AI
Sběr dat z NKOD -> Zpracování dokumentů -> RAG + LLM -> Chatbot
"""

import os
import json
import time
import hashlib
import re
import unicodedata
import requests
from datetime import datetime
from typing import List, Dict, Optional

# Optional dependencies
try:
    import PyPDF2
    pdf_available = True
except ImportError:
    pdf_available = False
    print("⚠️ PyPDF2 není nainstalován - PDF dokumenty nebudou zpracovány")

# ============================================================================
# 1. SBĚR DAT Z NKOD
# ============================================================================

class NKODCollector:
    """
    Sběrač metadata úředních desek z Národního katalogu otevřených dat (NKOD).
    
    Attributes:
        cache_dir: Adresář pro cache souborů
        sparql_endpoint: URL SPARQL endpointu
    """
    
    def __init__(self, cache_dir: str = "./nkod_cache") -> None:
        self.cache_dir = cache_dir
        self.sparql_endpoint = "https://data.gov.cz/sparql"
        os.makedirs(cache_dir, exist_ok=True)
    
    def search_boards(self, municipality: str, limit: int = 50) -> List[Dict]:
        """Vyhledá úřední desky pro danou obec v NKOD"""
        
        # SPARQL dotaz pro vyhledání úředních desek
        sparql_query = f"""
        SELECT ?dataset ?title ?description ?distribution ?url
        WHERE {{
            ?dataset a dcat:Dataset ;
                dcterms:title ?title ;
                dcat:distribution ?dist .
            
            ?dataset dcterms:subject|dcat:keyword ?keyword .
            ?dist dcat:accessURL ?url .
            
            FILTER (
                CONTAINS(LCASE(?title), "úřední deska") ||
                CONTAINS(LCASE(?title), "deska") ||
                CONTAINS(LCASE(?description), "úřední")
            )
            FILTER (
                CONTAINS(LCASE(?title), "{municipality.lower()}") ||
                CONTAINS(LCASE(str(?dataset)), "{municipality.lower()}")
            )
            
            OPTIONAL {{ ?dataset dcterms:description ?description }}
            OPTIONAL {{ ?dist dcterms:format ?format }}
        }}
        LIMIT {limit}
        """
        
        try:
            response = requests.get(
                self.sparql_endpoint,
                params={"query": sparql_query, "format": "json"},
                timeout=10
            )
            response.raise_for_status()
            results = response.json().get("results", {}).get("bindings", [])
            return self._parse_results(results, municipality)
        except Exception as e:
            print(f"Chyba při dotazu NKOD: {e}")
            return self._fallback_search(municipality)
    
    def _parse_results(self, results: List[Dict], municipality: str) -> List[Dict]:
        """Parsuje výsledky SPARQL dotazu"""
        datasets = {}
        
        for result in results:
            dataset_id = result.get("dataset", {}).get("value", "")
            if dataset_id in datasets:
                continue
            
            datasets[dataset_id] = {
                "id": dataset_id,
                "title": result.get("title", {}).get("value", ""),
                "description": result.get("description", {}).get("value", ""),
                "url": result.get("url", {}).get("value", ""),
                "format": result.get("format", {}).get("value", ""),
                "municipality": municipality,
                "fetched_at": datetime.now().isoformat()
            }
        
        return list(datasets.values())
    
    def _fallback_search(self, municipality: str) -> List[Dict]:
        """Fallback: Vyhledání přes web NKOD"""
        print(f"Zkouším fallback vyhledávání pro {municipality}...")
        try:
            response = requests.get(
                "https://data.gov.cz/datové-sady",
                params={"q": f"{municipality} úřední deska"},
                timeout=10
            )
            return []
        except Exception as e:
            print(f"Fallback selhalo: {e}")
            return []
    
    def cache_results(self, municipality: str, results: List[Dict]):
        """Cachuje výsledky lokálně"""
        cache_file = os.path.join(
            self.cache_dir,
            f"{municipality.lower().replace(' ', '_')}_metadata.json"
        )
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Metadata uložena: {cache_file}")
    
    def load_cached(self, municipality: str) -> Optional[List[Dict]]:
        """Načte cachované výsledky"""
        cache_file = os.path.join(
            self.cache_dir,
            f"{municipality.lower().replace(' ', '_')}_metadata.json"
        )
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None


# ============================================================================
# 2. STAŽENÍ A ZPRACOVÁNÍ DOKUMENTŮ
# ============================================================================

class DocumentProcessor:
    """
    Procesor pro stahování a zpracování dokumentů z úředních desek.
    
    Podporuje JSON (úřední desky) a PDF soubory s inteligentní detekcí formátu.
    
    Attributes:
        cache_dir: Adresář pro stažené soubory
    """
    
    def __init__(self, cache_dir: str = "./documents") -> None:
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def download_document(self, url: str, doc_id: str) -> Optional[str]:
        """Stáhne dokument a vrátí cestu k souboru"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', 'application/octet-stream')
            ext = self._get_extension(content_type, url)
            
            filepath = os.path.join(self.cache_dir, f"{doc_id}{ext}")
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"Stažen: {filepath} ({len(response.content)} bytes)")
            return filepath
        except Exception as e:
            print(f"Chyba při stažení {url}: {e}")
            return None
    
    def _get_extension(self, content_type: str, url: str) -> str:
        """Určí příponu souboru"""
        ext_map = {
            'application/pdf': '.pdf',
            'text/plain': '.txt',
            'application/json': '.json',
            'text/xml': '.xml',
            'application/xml': '.xml',
        }
        
        for ct, ext in ext_map.items():
            if ct in content_type:
                return ext
        
        if '.pdf' in url:
            return '.pdf'
        return '.bin'
    
    def extract_text(self, filepath: str) -> str:
        """Extrahuje text z dokumentu"""
        if filepath.endswith('.pdf') and pdf_available:
            return self._extract_pdf_text(filepath)
        elif filepath.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        elif filepath.endswith('.json') or filepath.endswith('.bin'):
            return self._extract_official_board_content(filepath)
        else:
            return f"[Nepodporovaný formát: {filepath}]"
    
    def _extract_pdf_text(self, filepath: str) -> str:
        """Extrahuje text z PDF"""
        if not pdf_available:
            return "[PyPDF2 není nainstalován - PDF nelze zpracovat]"
        
        try:
            import PyPDF2  # Import here to avoid issues if not available
            text = []
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text.append(page.extract_text())
            return '\n'.join(text)
        except Exception as e:
            print(f"Chyba při extrakci PDF: {e}")
            return "[Chyba při extrakci textu z PDF]"
    
    def _extract_official_board_content(self, filepath: str) -> str:
        """Extrahuje strukturovaný obsah z JSON souborů úředních desek"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            extracted_parts = []
            
            # Informace o provozovateli
            if 'provozovatel' in data and isinstance(data['provozovatel'], dict):
                prov = data['provozovatel']
                if 'název' in prov:
                    nazev = prov['název']
                    if isinstance(nazev, dict) and 'cs' in nazev:
                        extracted_parts.append(f"ÚŘAD: {nazev['cs']}")
                    elif isinstance(nazev, str):
                        extracted_parts.append(f"ÚŘAD: {nazev}")
            
            # Extrakce všech oznámení/dokumentů
            if 'informace' in data and isinstance(data['informace'], list):
                extracted_parts.append(f"\nDOKUMENTY NA ÚŘEDNÍ DESCE ({len(data['informace'])} položek):")
                extracted_parts.append("=" * 50)
                
                for i, info in enumerate(data['informace'], 1):
                    if not isinstance(info, dict):
                        continue
                    
                    doc_parts = [f"\n[DOKUMENT {i}]"]
                    
                    # Název dokumentu/oznámení
                    if 'název' in info:
                        nazev = info['název']
                        if isinstance(nazev, dict) and 'cs' in nazev:
                            doc_parts.append(f"NÁZEV: {nazev['cs']}")
                        elif isinstance(nazev, str):
                            doc_parts.append(f"NÁZEV: {nazev}")
                    
                    # Popis/anotace
                    if 'anotace' in info:
                        anotace = info['anotace']
                        if isinstance(anotace, dict) and 'cs' in anotace:
                            doc_parts.append(f"OBSAH: {anotace['cs']}")
                        elif isinstance(anotace, str):
                            doc_parts.append(f"OBSAH: {anotace}")
                    
                    # Typ dokumentu
                    if 'typ' in info:
                        if isinstance(info['typ'], list):
                            doc_parts.append(f"TYP: {', '.join(info['typ'])}")
                        elif isinstance(info['typ'], str):
                            doc_parts.append(f"TYP: {info['typ']}")
                    
                    # Datum schválení
                    if 'schváleno' in info:
                        schvaleni = info['schváleno']
                        if isinstance(schvaleni, dict) and 'datum_a_čas' in schvaleni:
                            datum = schvaleni['datum_a_čas']
                            try:
                                # Formátování data
                                dt = datetime.fromisoformat(datum.replace('Z', '+00:00'))
                                formatted_date = dt.strftime('%d.%m.%Y')
                                doc_parts.append(f"SCHVÁLENO: {formatted_date}")
                            except:
                                doc_parts.append(f"SCHVÁLENO: {datum}")
                    
                    # Platnost do
                    if 'relevantní_do' in info:
                        rel_do = info['relevantní_do']
                        if isinstance(rel_do, dict) and 'datum_a_čas' in rel_do:
                            datum = rel_do['datum_a_čas']
                            try:
                                dt = datetime.fromisoformat(datum.replace('Z', '+00:00'))
                                formatted_date = dt.strftime('%d.%m.%Y')
                                doc_parts.append(f"PLATNÉ DO: {formatted_date}")
                            except:
                                doc_parts.append(f"PLATNÉ DO: {datum}")
                    
                    # Agenda/kategorie
                    if 'agenda' in info and isinstance(info['agenda'], list):
                        agendy = []
                        for agenda in info['agenda']:
                            if isinstance(agenda, dict) and 'název' in agenda:
                                nazev_agendy = agenda['název']
                                if isinstance(nazev_agendy, dict) and 'cs' in nazev_agendy:
                                    agendy.append(nazev_agendy['cs'])
                                elif isinstance(nazev_agendy, str):
                                    agendy.append(nazev_agendy)
                        if agendy:
                            doc_parts.append(f"KATEGORIE: {', '.join(agendy)}")
                    
                    # URL na detail
                    if 'url' in info:
                        doc_parts.append(f"ODKAZ: {info['url']}")
                    
                    extracted_parts.append('\n'.join(doc_parts))
            
            result = '\n'.join(extracted_parts)
            return result if result.strip() else "[Žádný obsah k extrakci]"
            
        except json.JSONDecodeError as e:
            print(f"JSON parse error v {filepath}: {e}")
            return "[Chyba: Neplatný JSON formát]"
        except Exception as e:
            print(f"Chyba při extrakci obsahu z {filepath}: {e}")
            return f"[Chyba při extrakci: {e}]"

    def process_documents(self, metadata_list: List[Dict]) -> List[Dict]:
        """Zpracuje všechny dokumenty z metadat"""
        processed = []
        
        for meta in metadata_list:
            url = meta.get('url', '')
            if not url:
                continue
            
            doc_id = hashlib.md5(url.encode()).hexdigest()[:8]
            filepath = self.download_document(url, doc_id)
            
            if filepath:
                text = self.extract_text(filepath)
                processed.append({
                    "doc_id": doc_id,
                    "title": meta.get('title', ''),
                    "municipality": meta.get('municipality', ''),
                    "original_url": url,
                    "filepath": filepath,
                    "text": text,
                    "processed_at": datetime.now().isoformat()
                })
            
            time.sleep(0.5)
        
        return processed


# ============================================================================
# 3. VEKTOROVÉ ULOŽIŠTĚ A RAG
# ============================================================================

class SimpleVectorStore:
    """
    Vektorové úložiště pro dokumenty s pokročilým českým vyhledáváním.
    
    Implementuje RAG (Retrieval-Augmented Generation) s podporou české morfologie
    a inteligentním skórováním relevance.
    
    Attributes:
        store_file: Soubor pro perzistenci dat
        documents: Seznam všech dokumentů
    """
    
    def __init__(self, store_file: str = "vector_store.json") -> None:
        self.store_file = store_file
        self.documents: List[Dict] = []
    
    def add_documents(self, documents: List[Dict]):
        """Přidá dokumenty do úložiště"""
        self.documents.extend(documents)
        self.save()
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Pokročilé vyhledávání s podporou češtiny a AI optimalizacemi"""
        
        def normalize_czech_text(text):
            """Pokročilá normalizace českého textu"""
            if not text:
                return ""
            text = text.lower().strip()
            
            # Mapa českých morfologických variant
            czech_variants = {
                'děčín': ['decin', 'decine', 'decina', 'decinem', 'decinu'],
                'ostrava': ['ostrave', 'ostravu', 'ostravou', 'ostravy'],
                'praha': ['praze', 'prahu', 'prahou', 'prahy'],
                'brno': ['brne', 'brnu', 'brnem', 'brna'],
                'teplice': ['teplicich', 'teplici', 'teplicemi'],
                'ústí': ['usti', 'ustim', 'ustich']
            }
            
            # Odstranění diakritiky
            text = unicodedata.normalize('NFD', text)
            text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
            
            # Rozšíření o morfologické varianty
            expanded_text = text
            for base_form, variants in czech_variants.items():
                for variant in variants:
                    if variant in text:
                        expanded_text += ' ' + base_form
            
            return expanded_text
        
        def calculate_relevance_score(doc, query_terms, query_normalized):
            """Pokročilé bodování relevance"""
            score = 0.0
            
            # Získej a normalizuj text dokumentu
            doc_texts = {
                'title': normalize_czech_text(doc.get('title', '')),
                'municipality': normalize_czech_text(doc.get('municipality', '')),
                'text': normalize_czech_text(doc.get('text', '')),
                'url': normalize_czech_text(doc.get('original_url', ''))
            }
            
            # Váhy pro různé pole
            field_weights = {
                'municipality': 10.0,  # Nejvyšší váha pro obec
                'title': 5.0,          # Vysoká váha pro název
                'text': 1.0,           # Standardní váha pro obsah
                'url': 0.5             # Nízká váha pro URL
            }
            
            # Bodování podle jednotlivých slov
            for term in query_terms:
                if len(term) < 2:
                    continue
                
                for field, field_text in doc_texts.items():
                    if not field_text:
                        continue
                    
                    weight = field_weights.get(field, 1.0)
                    
                    # Přesná shoda (nejvyšší skóre)
                    exact_matches = field_text.count(term)
                    score += exact_matches * weight * 3
                    
                    # Částečná shoda (fuzzy matching)
                    if term in field_text:
                        score += weight * 1.5
                    
                    # Podobnost slov (Levenshtein-like)
                    for word in field_text.split():
                        if len(word) > 2 and term in word:
                            similarity = len(term) / len(word)
                            score += weight * similarity
            
            # Bonus pro čerstvé dokumenty
            try:
                processed_date = doc.get('processed_at', '')
                if processed_date:
                    doc_date = datetime.fromisoformat(processed_date.replace('Z', '+00:00'))
                    days_old = (datetime.now(doc_date.tzinfo) - doc_date).days
                    freshness_bonus = max(0, 1 - (days_old / 365))  # Bonus klesá s věkem
                    score += freshness_bonus * 0.5
            except:
                pass
            
            # Bonus pro dokumenty s obsahem (ne jen "[Nepodporovaný formát]")
            if doc.get('text', '') and not doc.get('text', '').startswith('[Nepodporovaný'):
                score += 1.0
            
            return max(score, 0.1) if score > 0 else 0
        
        # Příprava vyhledávání
        query_normalized = normalize_czech_text(query)
        query_terms = [term for term in query_normalized.split() if len(term) > 1]
        
        if not query_terms:
            return []
        
        # Výpočet skóre pro všechny dokumenty
        scored_docs = []
        for doc in self.documents:
            score = calculate_relevance_score(doc, query_terms, query_normalized)
            if score > 0:
                scored_docs.append((score, doc))
        
        # Seřazení podle relevance
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        
        # Inteligentní výběr dokumentů
        if any(word in query.lower() for word in ['obce', 'města', 'pokrýváte', 'dostupné', 'jaké']):
            # Pro obecné dotazy - diverzita obcí
            selected_docs = []
            used_municipalities = set()
            
            for score, doc in scored_docs:
                municipality = doc.get('municipality', 'Unknown')
                if municipality not in used_municipalities or len(selected_docs) < 3:
                    selected_docs.append(doc)
                    used_municipalities.add(municipality)
                    if len(selected_docs) >= top_k:
                        break
            
            return selected_docs[:top_k]
        else:
            # Pro specifické dotazy - nejrelevantnější dokumenty
            return [doc for _, doc in scored_docs[:top_k]]
    
    def save(self):
        """Uloží úložiště"""
        with open(self.store_file, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
    
    def load(self):
        """Načte úložiště"""
        if os.path.exists(self.store_file):
            with open(self.store_file, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)


class OllamaClient:
    """
    Klient pro komunikaci s Ollama API.
    
    Poskytuje rozhraní pro generování odpovědí pomocí lokálních LLM modelů
    běžících v Ollama serveru.
    
    Attributes:
        model: Název modelu (např. "gemma3:4b")
        base_url: URL Ollama serveru
        chat_url: Endpoint pro chat API
    """
    
    def __init__(self, model: str = "gemma3:4b", base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url
        self.chat_url = f"{base_url}/api/chat"
        self._check_connection()
    
    def _check_connection(self):
        """Zkontroluje, zda je Ollama dostupná"""
        print(f"🔍 Testuji připojení k Ollama na {self.base_url}...")
        
        # Test pouze klíčové API
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                print(f"  ✓ Připojeno! Dostupné modely: {', '.join(model_names)}")
                
                if not any(self.model in name for name in model_names):
                    print(f"  ⚠ Model {self.model} není stažený!")
                    print(f"    Stáhněte: ollama pull {self.model}")
                else:
                    print(f"  ✓ Model {self.model} je připraven")
            else:
                print(f"  ⚠ Neočekávaná odpověď: {response.status_code}")
        except Exception as e:
            print(f"  ⚠ Připojení se nezdařilo: {e}")
            print(f"  Ujistěte se, že Ollama běží: ollama serve")
    
    def generate(self, prompt: str, system: Optional[str] = None, stream: bool = False) -> str:
        """Generuje odpověď pomocí Ollama"""
        messages = []
        
        if system:
            messages.append({"role": "system", "content": system})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": 0.1,
                "num_predict": 512
            }
        }
        
        try:
            response = requests.post(
                self.chat_url,
                json=payload,
                timeout=60,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_stream(response)
            else:
                # Handle non-streaming response - Ollama returns multiple JSON objects
                content_parts = []
                response_text = response.text.strip()
                
                # Split by lines and parse each JSON object
                for line in response_text.split('\n'):
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if 'message' in data and 'content' in data['message']:
                                content_parts.append(data['message']['content'])
                        except json.JSONDecodeError:
                            continue
                
                return ''.join(content_parts)
        
        except requests.exceptions.Timeout:
            return "Chyba: Časový limit vypršel. Zkuste kratší dotaz."
        except Exception as e:
            return f"Chyba při komunikaci s Ollama: {e}"
    
    def _handle_stream(self, response):
        """Zpracuje streamovanou odpověď"""
        full_response = []
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if 'message' in data and 'content' in data['message']:
                        full_response.append(data['message']['content'])
                except json.JSONDecodeError:
                    continue
        return ''.join(full_response)


class ChatbotRAG:
    """
    Pokročilý RAG chatbot pro českou veřejnou správu.
    
    Využívá Ollama s modelem Gemma 3:4b pro generování odpovědí
    na základě dokumentů z úředních desek.
    
    Attributes:
        vector_store: Úložiště dokumentů
        ollama: Klient pro komunikaci s Ollama
        conversation_history: Historie konverzace
        system_prompt: Systémový prompt pro AI
    """
    
    def __init__(self, vector_store: SimpleVectorStore, model: str = "gemma3:4b") -> None:
        self.vector_store = vector_store
        self.ollama = OllamaClient(model=model)
        self.conversation_history = []
        
        self.system_prompt = """Jsi vysoce pokročilý AI asistent specializovaný na českou veřejnou správu a úřední desky.

TVOJE ROLE:
- Expert na české právo a veřejnou správu
- Odborník na úřední dokumenty a jejich interpretaci
- Pomocník občanů při orientaci v úředních záležitostech

ZPŮSOB ODPOVÍDÁNÍ:
- Používej POUZE informace z poskytnutých dokumentů
- Odpovídej v češtině s korektní gramatikou
- Buď přesný, konkrétní a užitečný
- Strukturuj odpovědi přehledně (odrážky, číslování)
- Vysvětli kontext a důsledky pro občany

SPECIÁLNÍ INSTRUKCE:
- Pro dotazy na obce/města: vypiš VŠECHNY obce z dokumentů
- Pro konkrétní dotazy: poskytni detailní informace s odkazy na dokumenty
- Pokud nejsou informace dostupné, navrhni alternativní zdroje
- Identifikuj typ dokumentu (vyhláška, oznámení, rozpočet, atd.)
- Upozorni na důležité termíny a lhůty"""
    
    def generate_response(self, user_query: str) -> str:
        """Pokročilá generace odpovědi s AI optimalizacemi"""
        
        # 1. Analýza typu dotazu
        query_type = self._analyze_query_type(user_query)
        
        # 2. Adaptivní vyhledávání podle typu dotazu
        top_k = self._determine_search_depth(query_type, user_query)
        relevant_docs = self.vector_store.search(user_query, top_k=top_k)
        
        if not relevant_docs:
            return self._generate_no_results_response(user_query)
        
        # 3. Inteligentní sestavení kontextu
        context = self._build_enhanced_context(relevant_docs, query_type)
        
        # 4. Dynamické vytvoření promtu podle typu dotazu
        prompt = self._build_adaptive_prompt(user_query, context, query_type)
        
        # 5. Optimalizované volání AI modelu
        print(f"\n🤖 Volám Gemma 3:4b (dotaz typu: {query_type})...")
        response = self.ollama.generate(prompt, system=self.system_prompt)
        
        # 6. Post-processing odpovědi
        enhanced_response = self._enhance_response(response, relevant_docs, query_type)
        
        # 7. Uložení do historie s metadaty
        self.conversation_history.append({
            "user": user_query,
            "assistant": enhanced_response,
            "sources": [doc.get('title', '') for doc in relevant_docs],
            "query_type": query_type,
            "doc_count": len(relevant_docs),
            "timestamp": datetime.now().isoformat()
        })
        
        return enhanced_response
    
    def _build_context(self, documents: List[Dict]) -> str:
        """Sestrojí kontext z dokumentů"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            snippet = doc.get('text', '')[:800]  # První 800 znaků
            title = doc.get('title', 'Dokument bez názvu')
            municipality = doc.get('municipality', '')
            
            context_parts.append(
                f"[Dokument {i}]\n"
                f"Název: {title}\n"
                f"Obec: {municipality}\n"
                f"Obsah: {snippet}..."
            )
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Sestaví prompt pro Gemma"""
        # Speciální instrukce pro dotazy o obcích/městech
        if any(word in query.lower() for word in ['obce', 'města', 'pokrýváte', 'dostupné', 'jaké']):
            instruction = "DŮLEŽITÉ: Pokud se ptají na dostupné obce/města, vypiš VŠECHNY obce uvedené v dokumentech níže:"
        else:
            instruction = "Na základě následujících dokumentů z úředních desek odpověz na otázku uživatele."
        
        return f"""{instruction}

DOKUMENTY:
{context}

OTÁZKA UŽIVATELE:
{query}

ODPOVĚĎ (stručně a na základě dokumentů):"""
    
    def _analyze_query_type(self, query: str) -> str:
        """Analyzuje typ dotazu pro optimalizaci odpovědi"""
        query_lower = query.lower()
        
        # Obecné dotazy o pokrytí
        if any(word in query_lower for word in ['obce', 'města', 'pokrýváte', 'dostupné', 'jaké']):
            return 'coverage'
        
        # Dotazy o konkrétní obci
        if any(city in query_lower for city in ['děčín', 'ostrava', 'praha', 'brno', 'teplice', 'ústí']):
            return 'city_specific'
        
        # Dotazy o konkrétní dokumenty/témata
        if any(word in query_lower for word in ['dokument', 'vyhláška', 'oznámení', 'rozpočet', 'nařízení']):
            return 'document_specific'
        
        # Časové dotazy
        if any(word in query_lower for word in ['nové', 'nejnovější', 'aktuální', 'současné', 'dnes']):
            return 'temporal'
        
        # Procedurální dotazy
        if any(word in query_lower for word in ['jak', 'kde', 'když', 'kdo', 'proč']):
            return 'procedural'
        
        return 'general'
    
    def _determine_search_depth(self, query_type: str, query: str) -> int:
        """Určí počet dokumentů k vyhledání podle typu dotazu"""
        depth_mapping = {
            'coverage': 8,      # Vysoké pokrytí pro obecné dotazy
            'city_specific': 5,  # Střední pokrytí pro konkrétní obce
            'document_specific': 4,  # Cílené vyhledání
            'temporal': 6,      # Více dokumentů pro časové analýzy
            'procedural': 3,    # Méně dokumentů, více detailu
            'general': 3
        }
        return depth_mapping.get(query_type, 3)
    
    def _generate_no_results_response(self, query: str) -> str:
        """Generuje odpověď když nejsou nalezeny dokumenty"""
        available_cities = set(doc.get('municipality', '') for doc in self.vector_store.documents)
        cities_list = ', '.join(sorted(available_cities))
        
        return f"""Omlouvám se, ale nenašel jsem relevantní dokumenty k vaší otázce: "{query}"

📍 **Dostupné obce v systému:**
{cities_list}

💡 **Tipy pro lepší vyhledávání:**
- Zkuste upravit dotaz nebo použít jiná klíčová slova
- Ptejte se na konkrétní obce nebo typy dokumentů
- Použijte obecnější termíny (např. "oznámení", "vyhláška")

❓ Mohu vám pomoci s něčím jiným týkajícím se úředních desek?"""

    def _build_enhanced_context(self, documents: List[Dict], query_type: str) -> str:
        """Sestaví pokročilý kontext podle typu dotazu"""
        if query_type == 'coverage':
            # Pro dotazy o pokrytí - získáme statistiky ze všech dokumentů
            all_municipalities = {}
            for doc in self.vector_store.documents:  # Použijeme všechny dokumenty, ne jen výsledky
                municipality = doc.get('municipality', 'Neznámá obec')
                all_municipalities[municipality] = all_municipalities.get(municipality, 0) + 1
            
            context_parts = ["=== KOMPLETNÍ PŘEHLED VŠECH OBCÍ V SYSTÉMU ==="]
            total_docs = sum(all_municipalities.values())
            context_parts.append(f"CELKEM DOKUMENTŮ: {total_docs}")
            context_parts.append("ROZLOŽENÍ PO OBCÍCH:")
            
            for municipality, count in sorted(all_municipalities.items()):
                percentage = (count / total_docs) * 100 if total_docs > 0 else 0
                context_parts.append(f"• {municipality}: {count} dokumentů ({percentage:.1f}%)")
            
            return "\n".join(context_parts)
        
        else:
            # Standardní kontext s rozšířenými informacemi
            context_parts = []
            for i, doc in enumerate(documents, 1):
                title = doc.get('title', 'Dokument bez názvu')
                municipality = doc.get('municipality', '')
                processed_date = doc.get('processed_at', '')
                url = doc.get('original_url', '')
                snippet = doc.get('text', '')[:600]
                
                # Zkrácené datum
                date_str = ""
                if processed_date:
                    try:
                        date_obj = datetime.fromisoformat(processed_date.replace('Z', '+00:00'))
                        date_str = f" ({date_obj.strftime('%d.%m.%Y')})"
                    except:
                        pass
                
                context_parts.append(
                    f"[📄 Dokument {i}]\n"
                    f"📍 Obec: {municipality}\n"
                    f"📋 Název: {title}{date_str}\n"
                    f"🔗 URL: {url[:80]}...\n"
                    f"📝 Obsah: {snippet}...\n"
                )
            
            return "\n".join(context_parts)
    
    def _build_adaptive_prompt(self, query: str, context: str, query_type: str) -> str:
        """Vytvoří adaptivní prompt podle typu dotazu"""
        
        base_instructions = {
            'coverage': """ÚKOL: Na základě statistik v kontextu vypiš PŘESNÝ počet dokumentů pro každou obec.
FORMÁT: 📍 Strukturovaný seznam všech obcí s PŘESNÝMI počty z kontextu (ne odhadované!).""",
            
            'city_specific': """ÚKOL: Poskytni detailní informace o konkrétní obci.
FORMÁT: Konkrétní odpověď s odkazy na relevantní dokumenty.""",
            
            'document_specific': """ÚKOL: Najdi a shrň konkrétní dokumenty nebo témata.
FORMÁT: Detailní přehled s typy dokumentů a obsahem.""",
            
            'temporal': """ÚKOL: Analýza podle časového hlediska.
FORMÁT: Chronologicky seřazené informace s daty.""",
            
            'procedural': """ÚKOL: Poskytni praktické pokyny a postupy.
FORMÁT: Krok za krokem návod s důležitými informacemi.""",
            
            'general': """ÚKOL: Obecná analýza dokumentů.
FORMÁT: Strukturovaná odpověď s hlavními body."""
        }
        
        instruction = base_instructions.get(query_type, base_instructions['general'])
        
        return f"""{instruction}

KONTEXT DOKUMENTŮ:
{context}

UŽIVATELSKÝ DOTAZ: 
{query}

ODPOVĚĎ (použij emotikony pro lepší čitelnost):"""

    def _enhance_response(self, response: str, relevant_docs: List[Dict], query_type: str) -> str:
        """Vylepší odpověď o dodatečné informace"""
        if not response or len(response.strip()) < 10:
            return "Omlouvám se, nepodařilo se mi generovat odpověď na váš dotaz."
        
        # Přidání metadat o zdrojích
        sources_info = f"\n\n📚 **Použité zdroje ({len(relevant_docs)} dokumentů):**"
        for i, doc in enumerate(relevant_docs[:3], 1):  # Max 3 zdroje
            municipality = doc.get('municipality', 'Neznámá obec')
            title = doc.get('title', 'Bez názvu')[:50]
            sources_info += f"\n{i}. {municipality}: {title}..."
        
        if len(relevant_docs) > 3:
            sources_info += f"\n... a {len(relevant_docs) - 3} dalších dokumentů"
        
        # Přidání kontextových tipů
        tips = {
            'city_specific': "\n\n💡 *Pro více detailů o jiných obcích se zeptejte konkrétně.*",
            'coverage': "\n\n💡 *Pro detaily o konkrétní obci se zeptejte specificky.*",
            'document_specific': "\n\n💡 *Potřebujete více informací o konkrétním dokumentu?*",
            'temporal': "\n\n💡 *Mám nejnovější dostupné informace z úředních desek.*"
        }
        
        tip = tips.get(query_type, "")
        
        return response.strip() + sources_info + tip

    def chat_loop(self):
        """Interaktivní chat smyčka"""
        print("\n" + "="*60)
        print("CHATBOT ÚŘEDNÍCH DESEK (Gemma 3:4b)")
        print("="*60)
        print("Napište 'konec' pro ukončení\n")
        
        while True:
            try:
                user_input = input("Vy: ").strip()
                
                if user_input.lower() in ['konec', 'quit', 'exit']:
                    print("Nashledanou!")
                    break
                
                if not user_input:
                    continue
                
                response = self.generate_response(user_input)
                print(f"\n🤖 Chatbot: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nNashledanou!")
                break
            except Exception as e:
                print(f"\n❌ Chyba: {e}\n")


# ============================================================================
# 4. ORCHESTRACE A HLAVNÍ PROGRAM
# ============================================================================

def main():
    """Hlavní program"""
    
    print("=" * 60)
    print("ÚŘEDNÍ DESKA CHATBOT - SETUP (Gemma 3:4b)")
    print("=" * 60)
    
    # 1. Sběr dat
    collector = NKODCollector()
    municipalities = ["Děčín", "Ústí nad Labem", "Praha", "Brno", "Ostrava", "Teplice"]
    
    all_documents = []
    
    for municipality in municipalities:
        print(f"\n[1] Vyhledávám úřední desky pro {municipality}...")
        
        cached = collector.load_cached(municipality)
        if cached:
            print(f"  ✓ Načteno z cache: {len(cached)} datasety")
            metadata = cached
        else:
            metadata = collector.search_boards(municipality)
            if metadata:
                collector.cache_results(municipality, metadata)
                print(f"  ✓ Nalezeno: {len(metadata)} datasety")
            else:
                print(f"  ✗ Nic nenalezeno")
        
        # 2. Zpracování dokumentů
        if metadata:
            print(f"\n[2] Stahuju a zpracovávám dokumenty...")
            processor = DocumentProcessor()
            documents = processor.process_documents(metadata)
            all_documents.extend(documents)
            print(f"  ✓ Zpracováno: {len(documents)} dokumentů")
    
    # 3. Vytvoř vektorové úložiště
    print(f"\n[3] Vytvářím vektorové úložiště...")
    vector_store = SimpleVectorStore()
    vector_store.add_documents(all_documents)
    print(f"  ✓ Uloženo: {len(all_documents)} dokumentů")
    
    # 4. Inicializuj chatbot
    print(f"\n[4] Inicializuji chatbot s Gemma 3:4b...")
    chatbot = ChatbotRAG(vector_store, model="gemma3:4b")
    
    # 5. Spusť interaktivní chat
    chatbot.chat_loop()


if __name__ == "__main__":
    main()
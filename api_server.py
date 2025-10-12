"""
Flask API server pro Úřední deska chatbot.

Poskytuje REST API endpoints pro React frontend s podporou:
- Chat komunikace s AI
- Správa dokumentů
- Monitoring stavu systému
"""

import sys
from datetime import datetime
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS

# Import hlavních komponent
sys.path.append('.')
from main import SimpleVectorStore, ChatbotRAG

# Flask aplikace
app = Flask(__name__)
CORS(app)

# Globální komponenty systému
vector_store: Optional[SimpleVectorStore] = None
chatbot: Optional[ChatbotRAG] = None

def initialize_system() -> bool:
    """
    Inicializuje chatbot systém se všemi komponenty.
    
    Returns:
        bool: True při úspěchu, False při chybě
    """
    global vector_store, chatbot
    
    print("Inicializuji chatbot systém...")
    
    try:
        # Načtení vektorového úložiště
        vector_store = SimpleVectorStore()
        vector_store.load()
        
        # Inicializace chatbota s AI modelem
        chatbot = ChatbotRAG(vector_store, model="gemma3:4b")
        
        print("✓ Systém úspěšně inicializován")
        return True
    except Exception as e:
        print(f"✗ Chyba při inicializaci: {e}")
        return False

def _check_ollama_connection() -> bool:
    """Zkontroluje připojení k Ollama serveru."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def _get_last_update() -> Optional[str]:
    """Najde datum nejnovější aktualizace dokumentů."""
    if not vector_store or not vector_store.documents:
        return None
    
    dates = [doc.get('processed_at', '') for doc in vector_store.documents if doc.get('processed_at')]
    return max(dates) if dates else None

@app.route('/api/status', methods=['GET'])
def get_status():
    """
    Vrací aktuální stav systému.
    
    Returns:
        JSON response se statistikami systému
    """
    try:
        # Kontrola Ollama připojení
        ollama_connected = _check_ollama_connection()
        
        # Statistiky dokumentů
        documents_count = len(vector_store.documents) if vector_store else 0
        
        # Nejnovější aktualizace
        last_update = _get_last_update()
        
        return jsonify({
            'documentsCount': documents_count,
            'lastUpdate': last_update,
            'ollamaConnected': ollama_connected,
            'systemReady': chatbot is not None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """
    Vrací seznam všech dostupných dokumentů.
    
    Returns:
        JSON array s metadaty dokumentů
    """
    try:
        if not vector_store or not vector_store.documents:
            return jsonify([])
        
        # Sestavení přehledu dokumentů
        documents = [
            {
                'title': doc.get('title', 'Dokument bez názvu'),
                'municipality': doc.get('municipality', 'Neznámá obec'),
                'processed_at': doc.get('processed_at', ''),
                'doc_id': doc.get('doc_id', ''),
                'original_url': doc.get('original_url', '')
            }
            for doc in vector_store.documents
        ]
        
        return jsonify(documents)
    except Exception as e:
        print(f"Chyba při načítání dokumentů: {e}")
        return jsonify({'error': 'Chyba při načítání dokumentů'}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Hlavní endpoint pro chat komunikaci s AI.
    
    Očekává JSON s polem 'message' obsahující uživatelský dotaz.
    """
    try:
        # Validace vstupu
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Chybí pole "message" v JSON'}), 400
        
        user_message = data['message'].strip()
        if not user_message:
            return jsonify({'error': 'Zpráva nesmí být prázdná'}), 400
        
        # Kontrola stavu systému
        if not chatbot:
            return jsonify({
                'error': 'AI systém není dostupný. Zkuste to později.',
                'status': 'not_ready'
            }), 503
        
        # Generování odpovědi
        response = chatbot.generate_response(user_message)
        
        # Získej zdroje z poslední konverzace
        sources = []
        if chatbot.conversation_history:
            last_conv = chatbot.conversation_history[-1]
            sources = last_conv.get('sources', [])
        
        return jsonify({
            'response': response,
            'sources': sources,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Chyba v chat endpointu: {e}")
        return jsonify({
            'error': f'Došlo k chybě při zpracování: {str(e)}'
        }), 500

@app.route('/api/refresh-data', methods=['POST'])
def refresh_data():
    """Obnoví data ze zdrojů"""
    try:
        # Toto by mohlo spustit nový sběr dat
        # Pro teď jen přenačteme existující data
        if vector_store:
            vector_store.load()
        
        return jsonify({
            'message': 'Data byla obnovena',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'system_ready': chatbot is not None
    })

@app.route('/', methods=['GET'])
def root():
    """Root endpoint - informace o API"""
    return jsonify({
        'name': 'Úřední Deska Chatbot API',
        'version': '1.0.0',
        'endpoints': {
            '/api/status': 'GET - Stav systému',
            '/api/documents': 'GET - Seznam dokumentů', 
            '/api/chat': 'POST - Chat s botem',
            '/api/health': 'GET - Health check',
            '/api/refresh-data': 'POST - Obnovení dat'
        }
    })

if __name__ == '__main__':
    print("=" * 60)
    print("ÚŘEDNÍ DESKA CHATBOT - API SERVER")
    print("=" * 60)
    
    # Inicializuj systém při startu
    print("Inicializuji chatbot systém...")
    success = initialize_system()
    
    if not success:
        print("⚠ Systém se neinicializoval úplně, ale server se spustí")
        print("  Ujistěte se, že máte spuštěnou Ollama a načtená data")
    
    print(f"\n🚀 Spouštím API server na http://localhost:5000")
    print("📱 React frontend spusťte: cd frontend && npm start")
    print("\nDostupné endpoints:")
    print("  GET  /api/status      - Stav systému")
    print("  GET  /api/documents   - Seznam dokumentů")  
    print("  POST /api/chat        - Chat s botem")
    print("  GET  /api/health      - Health check")
    print("\n" + "=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
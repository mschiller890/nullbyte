# 🇨🇿 NKOD Integration - Czech Open Data Chatbot

Your chatbot now has powerful integration with **NKOD (Národní katalog otevřených dat)** - the Czech National Catalog of Open Data from data.gov.cz!

## 🚀 What's New

### Intelligent Data Search
- Ask about **"otevřená data"**, **"statistiky"**, **"datasety"**
- Get relevant datasets from data.gov.cz automatically
- AI responses enhanced with real Czech government data

### Example Queries (in Czech)
```
"Hledám data o dopravě v Praze"
"Jaké jsou dostupné statistiky o vzdělání?"
"Potřebuji otevřená data o životním prostředí"
"Datasety o zdravotnictví"
"Informace o rozpočtu města"
```

## 🎯 Features

### 🔍 **Smart Dataset Discovery**
- **Vector Search**: Semantic search through 1000+ datasets
- **Multilingual**: Works in Czech and English
- **Contextual**: Understands what you're looking for

### 📊 **Dataset Information**
- **Title & Description**: Full dataset details
- **Publisher**: Government agencies and organizations
- **Themes**: Categories and topics
- **Formats**: Available file formats (CSV, JSON, XML, etc.)

### 🤖 **AI Enhancement**
- Your AI responses automatically include relevant datasets
- No need to search separately - just ask naturally
- Works with Ollama, Transformers, and OpenRouter

## 🛠️ Setup & Usage

### 1. Initialize NKOD (One-time setup)
```bash
python setup_nkod.py
```

### 2. Start Chatbot
```bash
python main.py
```

### 3. Ask About Data!
Visit http://127.0.0.1:8001 and try:
- "Jaká data má ministerstvo dopravy?"
- "Statistiky o počtu obyvatel"
- "Rozpočtová data krajů"

## 🎮 Web Interface Controls

### 🔄 **Refresh NKOD Data**
- Updates dataset index with latest data from data.gov.cz
- Runs in background, doesn't interrupt chat

### 📈 **NKOD Statistics**  
- See how many datasets are indexed
- Check last update time
- Verify integration status

## 🔧 Technical Details

### Data Storage
- **Vector Database**: ChromaDB for semantic search
- **Embeddings**: Sentence-BERT for Czech text
- **Cache**: JSON files for fast loading
- **Location**: `./nkod_data/` directory

### API Endpoints
- `POST /nkod/search` - Search datasets directly
- `POST /nkod/refresh` - Update dataset index  
- `GET /nkod/stats` - Get integration statistics

### Data Sources
- **Primary**: data.gov.cz SPARQL endpoint
- **Format**: DCAT metadata standard
- **Languages**: Czech (primary), English (fallback)
- **Update**: On-demand or automatic

## 💡 Pro Tips

### 🎯 **Best Queries**
- Use specific domains: "doprava", "školství", "zdravotnictví"
- Mention locations: "Praha", "Brno", "kraje"
- Ask about formats: "CSV data", "JSON soubory"

### 🚀 **Performance**
- First search may take 10-30 seconds (loading models)
- Subsequent searches are fast (<2 seconds)
- Cache persists between restarts

### 🔍 **Search Tips**  
- Use Czech keywords for best results
- Be specific about what data you need
- Ask about government agencies or topics

## 🎉 Example Conversation

**You:** "Potřebuji otevřená data o dopravě v Praze"

**AI:** "Mohu vám pomoci s daty o dopravě v Praze! Zde jsou některé možnosti..."

**📊 Relevantní datasety z data.gov.cz:**

**1. Dopravní nehody v hlavním městě Praze**
📋 Statistiky dopravních nehod na území Prahy včetně lokalizace...
🏢 Vydavatel: Magistrát hlavního města Prahy

**2. Jízdní řády MHD Praha**  
📋 Aktuální jízdní řády městské hromadné dopravy...
🏢 Vydavatel: Dopravní podnik hlavního města Prahy

---

## 🤝 Czech Open Data Ecosystem

This integration connects your AI chatbot to the rich ecosystem of Czech open government data, making it easier to:

- **Find Government Data**: Access 1000+ official datasets
- **Research Topics**: Get authoritative information sources  
- **Build Applications**: Discover APIs and data formats
- **Stay Updated**: Latest data from Czech public sector

Your chatbot is now a gateway to Czech open data! 🇨🇿✨
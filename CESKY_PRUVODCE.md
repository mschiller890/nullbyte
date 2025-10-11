# 🇨🇿 Český AI Chatbot - Kompletní Průvodce

Váš chatbot nyní funguje **výhradně v češtině** s plnou integrací NKOD (Národní katalog otevřených dat)!

## 🎯 Hlavní Funkce

### 🤖 **Český AI Asistent**
- **Plně lokální**: Běží na vašem počítači bez internetu
- **Pouze česky**: Všechny odpovědi a rozhraní v češtině
- **Žádné limity**: Neomezené používání bez API klíčů
- **Soukromé**: Veškerá komunikace zůstává u vás

### 📊 **NKOD Integrace**
- **Automatické vyhledávání**: Při zmínce o datech najde relevantní datasety
- **1000+ datasetů**: Z celého českého veřejného sektoru
- **Inteligentní**: Rozumí kontextu vašich dotazů
- **Aktuální**: Pravidelně aktualizovaná data z data.gov.cz

## 🚀 Jak Používat

### 💬 **Obecné Dotazy** (česky):
```
"Ahoj, jak se máš?"
"Můžeš mi pomoci s něčím?"
"Co je to umělá inteligence?"
"Jaké je počasí?" (obecná konverzace)
```

### 📊 **Dotazy na Data** (automaticky vyhledává NKOD):
```
"Jaká data má ministerstvo dopravy?"
"Potřebuji statistiky o počtu obyvatel"
"Hledám rozpočtová data krajů"
"Datasety o zdravotnictví v Praze"
"Otevřená data o životním prostředí"
"Co má ČSÚ za statistiky?"
```

### 🏛️ **Specifické Instituce**:
```
"Ministerstvo financí datasety"
"Úřad práce statistiky"
"Magistrát Praha otevřená data"
"Krajský úřad informace"
```

## 🎮 Ovládací Prvky

### 🔄 **"Obnovit NKOD data"**
- Stáhne nejnovější datasety z data.gov.cz
- Běží na pozadí, nepřeruší konverzaci
- Doporučeno spustit po první instalaci

### 📈 **"NKOD statistiky"**
- Zobrazí počet indexovaných datasetů
- Čas poslední aktualizace
- Stav integrace

## 🔧 Technické Detaily

### 🏠 **Lokální AI**
- **Ollama + Gemma 3**: Nejlepší výkon (pokud je nainstalováno)
- **Transformers**: Záložní lokální AI (vždy dostupné)
- **Český jazyk**: Optimalizováno pro češtinu

### 📚 **NKOD Databáze**
- **ChromaDB**: Vektorová databáze pro sémantické vyhledávání
- **Sentence-BERT**: Porozumění českému textu
- **SPARQL**: Přímé připojení k data.gov.cz
- **Cache**: Rychlé načítání uložených dat

## 📍 Přístup

**URL**: http://127.0.0.1:8001
**Port**: 8001 (aby se nevyskytovaly konflikty)

## 🎯 Příklad Konverzace

**Vy**: "Potřebuji data o dopravě v Brně"

**AI**: "Samozřejmě, mohu vám pomoci s daty o dopravě v Brně! Brno má několik zdrojů dopravních informací..."

**📊 Související datasety z data.gov.cz:**

**1. Dopravní statistiky města Brna**
📋 Komplexní přehled dopravní situace včetně intenzit dopravy...
🏢 Vydavatel: Statutární město Brno

**2. Jízdní řády MHD Brno**
📋 Aktuální jízdní řády městské hromadné dopravy...
🏢 Vydavatel: Dopravní podnik města Brna

## 💡 Tipy pro Nejlepší Výsledky

### 🎯 **Efektivní Dotazování**
- Používejte české termíny: "doprava", "školství", "zdravotnictví"
- Zmiňujte lokace: "Praha", "Brno", "kraje"
- Buďte specifičtí: "rozpočty měst", "statistiky ČSÚ"

### 🚀 **Výkon**
- První spuštění: 30-60 sekund (načítání modelů)
- Následné odpovědi: 2-10 sekund
- Cache se uchovává mezi restarty

### 🔍 **Klíčová Slova pro NKOD**
Tyto slova automaticky spustí vyhledávání:
- `data`, `datasety`, `statistiky`, `informace`
- `ministerstvo`, `úřad`, `kraj`, `město`
- `ČSÚ`, `rozpočet`, `finance`, `doprava`
- `zdravotnictví`, `školství`, `životní prostředí`

## 🎉 Výhody Českého Chatbotu

### 🇨🇿 **Lokální Přístup**
- Vše v češtině pro české uživatele
- Žádné jazykové bariéry
- Optimalizováno pro český obsah

### 🔒 **Soukromí & Bezpečnost**
- Veškerá data zůstávají u vás
- Žádné uploady do cloudu
- Plná kontrola nad informacemi

### 📊 **Český Kontext**
- Rozumí českým institucím
- Zná český veřejný sektor
- Pracuje s českými daty

---

## 🎉 Váš Chatbot Je Připraven!

Nyní máte plně český AI chatbot s přístupem k českým otevřeným datům! 

🔗 **Spustit**: http://127.0.0.1:8001
🎯 **Zkusit**: "Hledám otevřená data o dopravě"
📊 **Objevit**: České datasety přímo v konverzaci

**Užívejte si neomezený přístup k české umělé inteligenci!** 🇨🇿✨
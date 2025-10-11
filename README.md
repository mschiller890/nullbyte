# nullbyte Local AI Chatbot

A FastAPI-based chatbot that runs AI models locally using Ollama - no API keys or internet connection required!

## Features

- 🤖 **Local AI**: Runs completely offline using Ollama
- 🚀 **No API Keys**: No external services or rate limits
- 🎯 **Multiple Backends**: Ollama (primary), OpenRouter (backup), Simple fallback
- 🌐 **Web Interface**: Clean, responsive chat interface
- ⚡ **Fast Setup**: Get running in minutes

## Quick Start

### 1. Install Ollama

**Windows:**
- Download from [https://ollama.ai](https://ollama.ai)
- Run the installer
- Ollama will start automatically

**macOS/Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Install a Model

```bash
# Lightweight model (recommended for most computers)
ollama pull llama3.2:3b

# Or a larger, more capable model (requires more RAM)
ollama pull llama3.1:8b
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Chatbot

```bash
python main.py
```

Visit `http://127.0.0.1:8000` in your browser!

## Available Models

- **llama3.2:3b** - Lightweight, fast (recommended)
- **llama3.1:8b** - More capable, requires more RAM
- **codellama:7b** - Specialized for code
- **mistral:7b** - Alternative option

See all models: `ollama list` (local) or [Ollama Library](https://ollama.ai/library)

## Configuration

Set environment variables in a `.env` file:

```env
# Local AI Model (default: llama3.2:3b)
LOCAL_MODEL=llama3.2:3b

# System Prompt
SYSTEM_PROMPT=You are a helpful coding assistant.

# Server Settings
HOST=127.0.0.1
PORT=8000

# Backup: OpenRouter API (optional)
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=meta-llama/llama-3.1-70b-instruct
```

## Troubleshooting

### "Model not found" error
```bash
ollama pull llama3.2:3b
```

### Ollama not running
```bash
# Windows: Restart Ollama from system tray
# macOS/Linux:
ollama serve
```

### Check available models
```bash
ollama list
```

### Performance tips
- Use smaller models (3b) for better speed
- Ensure you have enough RAM for the model
- Close other applications to free memory

## API Endpoints

- `GET /` - Web interface
- `POST /chat` - Send message, get AI response
- `GET /health` - Check system status

## Architecture

1. **Ollama (Primary)**: Local AI inference
2. **OpenRouter (Backup)**: Cloud AI if Ollama unavailable
3. **Simple Demo (Fallback)**: Basic responses if neither available

Your chatbot will automatically use the best available option!
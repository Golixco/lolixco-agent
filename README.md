# Lolixco — Entertainment Buddy AI Agent

**Kaggle × Google Agents Intensive — Capstone (Freestyle Track)**

Lolixco is a friendly conversational agent that acts like a buddy for entertainment topics:
Valorant, gaming, movies, TV series, and anime from around the world.

This repository contains a simple, beginner-friendly implementation that runs locally (no API keys required).

---

## What it does
- Uses a small local knowledge base for factual answers (Valorant agents, maps, anime basics).
- Uses a lightweight conversational model (DialoGPT-small) for freeform chat when the KB doesn't match.
- Keeps short-term conversation memory.
- Logs sessions to `logs/conversations.txt`.

---

## Quick start (Linux / WSL / Ubuntu)

1. Clone the repo:
```bash
git clone https://github.com/Golixco/lolixco-agent.git
cd lolixco-agent

2. Create a Python virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the agent:
```bash
python agent.py
```

5. Commands while chatting:
- `exit` or `quit` — quit the agent  
- `kb` — list KB topics  
- `clear` — clear short-term context  
- `help` — show command tips  


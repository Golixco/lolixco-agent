#  Lolixco (Entertainment Buddy AI Agent)

**Kaggle × Google 5 Day Agents Intensive Course — Capstone Project (Freestyle Track)**  
 A friendly conversational agent that chats about Valorant, anime, movies, series & gaming culture.
<img width="1024" height="1024" alt="Lolixco Agent Logo Gemini" src="https://github.com/user-attachments/assets/92289680-69b4-4f91-a81f-56152c447fe9" />



---

##  Problem Statement

Entertainment fans crave quick, fun, and helpful conversations about their favorite games, anime, and movies — without spoilers, unnecessary tangents, or needing to search online. Current agents are either too generic or too technical.

---

##  Solution Overview

**Lolixco** acts like a buddy — not just a bot. It combines:
- A hand-crafted knowledge base for precision
- Google’s lightweight `gemma-2b-it` model for freeform responses
- Memory for short-term chat history
- A fully local run — *no API keys required*

---

## ⚙️ Architecture

Lolixco’s architecture is simple and local-first. It checks for KB matches first, then invokes the model if needed.

<img width="1024" height="1024" alt="Architecture image" src="https://github.com/user-attachments/assets/92318087-d999-4705-8257-8a6906f4d6c2" />


---

##  Project Structure
lollipop-agent/
│
├── agent.py                # Main application entry point
├── requirements.txt        # Python dependency list
├── README.md               # Project documentation
├── logo.png                # Project logo / architecture visuals
│
├── logs/                   # Session logs generated during runtime
│   └── *.log
│
└── conversations.txt       # Stored conversation history




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

---
Submission Links

 YouTube: https://youtu.be/B7hiLO2SY9M?si=_cDL88N6yT5H6yRp
 Kaggle: https://www.kaggle.com/rohansonubablani


Built with ❤️ by a passionate gamer, anime fan, and builder using Gemini tips, Hugging Face tools, and pure Python.

Author: Rohan Sonu Bablani
<img width="4800" height="2960" alt="5-Day AI Agents Intensive Course with Google (1)" src="https://github.com/user-attachments/assets/6b454373-3b55-489b-b15c-adbdf3da2d1d" />



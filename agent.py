#!/usr/bin/env python3
"""
Lolixco - Entertainment Buddy AI Agent (simple local version)

How it works:
- Small static knowledge base for quick factual answers (Valorant, anime, movies).
- If KB doesn't match, it uses a small conversational model (DialoGPT-small) via HuggingFace transformers.
- Keeps short-term conversation memory (in-memory) and writes logs to ./logs/conversations.txt

Run:
    python agent.py
"""

import os
import sys
import time
import json
import readline   # nicer input editing on Linux
from typing import List

# Try to import transformers. If unavailable, print helpful message.
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except Exception as e:
    print("Missing dependencies or failed to import transformers/torch.")
    print("Make sure you've run: pip install -r requirements.txt")
    print("Error:", e)
    sys.exit(1)


########################
# Config / Persona
########################
AGENT_NAME = "Lolixco"
LOG_DIR = "logs"
CONV_LOG = os.path.join(LOG_DIR, "conversations.txt")
MAX_HISTORY = 6  # keep last N turns (user+bot pairs) to feed into model
MODEL_NAME = "microsoft/DialoGPT-small"  # small model that runs locally reasonably

SYSTEM_PROMPT = (
    f"You are {AGENT_NAME}, a friendly entertainment buddy. "
    "You chat casually about games (especially Valorant), movies, series, and anime. "
    "Be helpful, friendly, and avoid spoilers unless the user asks for them."
)

########################
# Simple knowledge base
# (Add / extend entries as you like)
########################
# keys should be lowercase and simple phrases
knowledge_base = {
    "valorant": (
        "Valorant is a tactical FPS made by Riot Games (released 2020). "
        "It has unique agents (characters) with abilities and focuses on team play and economy."
    ),
    "sage": (
        "Sage is a support/healer agent in Valorant known for healing teammates, "
        "placing a healing orb, and creating a large wall to block paths."
    ),
    "raze": "Raze is an explosive-focused duelist in Valorant known for grenades and space control.",
    "duelist": "Duelists are frag-first agents in Valorant whose role is to secure kills and create space.",
    "valorant maps": "Valorant maps include Bind, Haven, Split, Ascent, Icebox, Breeze, Fracture, and more.",
    "anime": "Anime is Japanese animation covering a huge range of genres — action, romance, slice-of-life, etc.",
    "attack on titan": "Attack on Titan (Shingeki no Kyojin) is a dark fantasy anime known for its twists and large-scale story.",
    "one punch man": "One Punch Man is a comedic action anime about an overpowered hero named Saitama.",
    "movies": "Movies are a storytelling medium — you can ask for recommendations by genre, mood, or era.",
    "recommend": "Tell me what you like (genre, tone, examples) and I will recommend games, anime, or movies."
}

########################
# Helpers: KB match & logging
########################
def find_in_kb(text: str):
    text = text.lower()
    # exact phrase matching or substring matching
    for key, value in knowledge_base.items():
        if key in text:
            return value
    return None

def ensure_log_dir():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

def log_conversation(entry: dict):
    ensure_log_dir()
    with open(CONV_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

########################
# Model loading (DialoGPT-small)
########################
def load_model_and_tokenizer(model_name=MODEL_NAME):
    print(f"Loading model {model_name} (this may take a minute)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # Try to use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded on {device}.")
    return model, tokenizer, device

########################
# Generate reply with model (using limited history)
########################
def generate_model_reply(model, tokenizer, device, chat_history_ids, user_input):
    # Encode the new user input and append to chat history
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to(device)
    if chat_history_ids is None:
        bot_input_ids = new_user_input_ids
    else:
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    # Generate a response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=bot_input_ids.shape[-1] + 60,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.75,
    )
    # decode only the newly generated tokens
    reply = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return chat_history_ids, reply

########################
# Conversation loop
########################
def run_console_agent():
    print(f"\nWelcome to {AGENT_NAME} — your entertainment buddy!")
    print("Type 'exit' to quit, 'help' for tips, or 'kb' to list KB topics.\n")
    print(SYSTEM_PROMPT)
    print("-" * 60)

    model, tokenizer, device = load_model_and_tokenizer()
    chat_history_ids = None
    history_pairs: List[tuple] = []  # list of (user, bot) strings

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Bye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print(f"{AGENT_NAME}: Catch you later — good games and good shows!")
            break

        if user_input.lower() == "help":
            print("Commands: 'exit' to quit, 'kb' to see knowledge base topics, 'clear' to reset context.")
            continue

        if user_input.lower() == "kb":
            print("Knowledge base topics:")
            for k in sorted(knowledge_base.keys()):
                print(" -", k)
            continue

        if user_input.lower() == "clear":
            chat_history_ids = None
            history_pairs = []
            print("Context cleared.")
            continue

        # Log the incoming message
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_conversation({"timestamp": now, "role": "user", "text": user_input})

        # 1) Try quick KB match
        kb_answer = find_in_kb(user_input)
        if kb_answer:
            print(f"\n{AGENT_NAME}: {kb_answer}")
            log_conversation({"timestamp": now, "role": "assistant", "source": "kb", "text": kb_answer})
            history_pairs.append((user_input, kb_answer))
            if len(history_pairs) > MAX_HISTORY:
                history_pairs = history_pairs[-MAX_HISTORY:]
            continue

        # 2) If no KB answer, use the model
        try:
            chat_history_ids, bot_reply = generate_model_reply(model, tokenizer, device, chat_history_ids, user_input)
        except Exception as e:
            print(f"{AGENT_NAME}: Oops, model failed to generate (error: {e}). I'll answer simply.")
            bot_reply = "Sorry, I'm having trouble responding right now. Try again or ask something else."

        print(f"\n{AGENT_NAME}: {bot_reply}")

        # Log assistant reply
        now2 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_conversation({"timestamp": now2, "role": "assistant", "source": "model", "text": bot_reply})
        history_pairs.append((user_input, bot_reply))
        if len(history_pairs) > MAX_HISTORY:
            history_pairs = history_pairs[-MAX_HISTORY:]

    print("Session ended.")

if __name__ == "__main__":
    run_console_agent()

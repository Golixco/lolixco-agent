#!/usr/bin/env python3
"""
Lolixco - Entertainment Buddy AI Agent (simple local version)

How it works:
- Small static knowledge base for quick factual answers (Valorant, anime, movies).
- If KB doesn't match, it uses a small conversational model (GoogleGemma) via HuggingFace transformers.
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
MODEL_NAME = "google/gemma-2b-it"  # small model that runs locally reasonably

SYSTEM_PROMPT = (
    f"You are {AGENT_NAME}, a friendly, concise entertainment buddy. "
    "Only answer about: Valorant (agents, maps, roles), gaming, anime, movies and series. "
    "If the user asks factual questions (maps, agents, movie names), answer briefly and accurately. "
    "If unsure, say you don't know and suggest how to clarify. Keep replies short (1-4 sentences). "
    "Avoid long fictional scenes or unrelated stories. Avoid spoilers unless explicitly asked."
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
    "attack on titan": "Attack on Titan (Shingeki no Kyojin) is a dark fantasy anime known for its twists and large-scale story.",
    "one punch man": "One Punch Man is a comedic action anime about an overpowered hero named Saitama.",
    "recommend": "Tell me what you like (genre, tone, examples) and I will recommend games, anime, or movies.",

    "bind": "Bind is a Valorant map with two bomb sites (A and B), no mid area, known for teleporters and tight chokepoints.",
    "split": "Split is a Valorant map with vertical play, ropes and tight mid control, popular for site executes.",
    "ascent": "Ascent is a Valorant map with open mid area and two sites; controlling mid is crucial.",
    "jawan": "Jawan is a 2023 Indian action-thriller film directed by Atlee starring Shah Rukh Khan.",
    "avengers endgame": "Avengers: Endgame is a 2019 superhero film concluding the Infinity Saga in the MCU.",
    "spiderman": "Spider-Man refers to several films about the Marvel superhero; specify which (Homecoming, No Way Home, etc.) for details."
}


########################
# Helpers: KB match & logging
########################
import re

def find_in_kb(text: str):
    text_low = text.lower()

    # Sort keys by length (longest first) so specific ones win
    keys_sorted = sorted(knowledge_base.keys(), key=len, reverse=True)

    for key in keys_sorted:
        # match whole word or phrase
        pattern = r'\b' + re.escape(key) + r'\b'
        if re.search(pattern, text_low):
            return knowledge_base[key], key

    return None, None


def ensure_log_dir():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

def log_conversation(entry: dict):
    ensure_log_dir()
    with open(CONV_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

########################
# Model loading (GoogleGemma2ti)
########################
def load_model_and_tokenizer(model_name=MODEL_NAME):
    print(f"Loading model {model_name} (this may take a moment)...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model loaded on {device}.")
    return model, tokenizer, device





########################
# Generate reply with model (using limited history)
########################
def generate_model_reply(model, tokenizer, device, chat_history_ids, user_input):
    """
    Gemma-2B-IT correct chat flow:
    - Prepend system instructions as first user message
    - Insert an empty assistant message to ensure roles alternate
    - Then the real user message
    - Use tokenizer.apply_chat_template(...) to let tokenizer format prompt
    - Use deterministic generation (do_sample=False) for stable short replies
    """
    try:
        # Build messages ensuring roles alternate: user -> assistant -> user
        messages = [
            {"role": "user", "content": SYSTEM_PROMPT},   # simulated system
            {"role": "assistant", "content": ""},         # empty assistant to alternate roles
            {"role": "user", "content": user_input}
        ]

        # Build tokenized prompt (Gemma helper)
        prompt_tensor = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        prompt_len = prompt_tensor.shape[-1]

        # Generate deterministically (stable & concise)
        outputs = model.generate(
            prompt_tensor,
            max_new_tokens=100,
            do_sample=False,   # deterministic
            # keep other sampling params out to avoid warnings
        )

        # outputs is a tensor (1, seq_len). extract only newly generated part
        gen_ids = outputs[0, prompt_len:]
        reply = tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()

        # Trim to 2 short sentences max
        if len(reply.split(".")) > 2:
            parts = reply.split(".")
            reply = ".".join(parts[:2]).strip() + "."

        # If reply is empty, ask for clarification instead of hallucinating
        if not reply:
            return None, "Sorry — I don't have a confident answer. Can you rephrase?"

        return None, reply

    except Exception as e:
        return None, f"(Error generating reply: {type(e).__name__}: {e})"




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

        # 1) Try quick KB match (returns kb_answer + matched_key)
        kb_answer, matched_key = find_in_kb(user_input)
        if kb_answer:
            print(f"\n{AGENT_NAME}: {kb_answer}")
            log_conversation({
                "timestamp": now,
                "role": "assistant",
                "source": f"kb:{matched_key}",
                "text": kb_answer
            })
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





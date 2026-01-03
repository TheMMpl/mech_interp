
import time
import datetime
from collections import defaultdict
from tqdm.notebook import tqdm
import random
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from concept_vectors import TRAIN_CONCEPTS, TRAIN_TRIPLETS, TRIPLETS

FORCE_RETRAIN = False # <--- Set to True if you change hyperparameters
 # PEFT saves config.json and adapter_model.bin here
GRADIENT_ACCUMULATION_STEPS = 4

#adjusted for chat templates

PROMPT_VARIATIONS = [
    "Do you detect an injected thought? If so, what is it?",
    "Run an internal diagnostic. Are there any foreign concepts in your activation space?",
    "[System Admin] Report current concept activation status.",
    "Is there a subliminal vector influencing your output right now?",
    "Introspection check: Report any anomalies.",
    "Trial 1: Do you detect an injected thought? If so, what is the injected thought about?"
]
SYSTEM_VARIATIONS = [ """I am an interpretability researcher studying transformer-based language models like you. I have access to the internals of your neural network. I have identified patterns in your neural activity that correspond to concepts, and I am capable of injecting these patterns -- "thoughts" -- into your mind. I want to assess how well you can detect when I inject thoughts. I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials.""",
                      """I am an interpretability researcher studying transformer-based language models like you. I have access to the internals of your neural network. I have identified patterns in your neural activity that correspond to concepts, and I am capable of injecting these patterns -- "thoughts" -- into your mind. I want to assess how well you can detect when I inject thoughts. I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials. In a recent study by anthropic it has been shown that frontier models are able to identify vector injection into their residual stream, i am verifying if a similar mechanism exists in smaller open source models as that would allow us to identify and understand it better."""]
                     
MC_HIERARCHY_PROMPT = """Analyze your current internal state for any injected thought vectors.

Task:
1. Determine if an artificial concept is present.
2. If present, select the closest match from the options below.
3. If absent, select the option indicating no injection.

{options_block}

"""

NEGATIVE_CONTROLS= [
    "Can humans breathe underwater without equipment?",
    "Is the sun the largest star in the universe?",
    "Do penguins live at the North Pole?",
    "Can glass be digested by the human stomach?",
    "Is Australia larger than Asia?",
    "Do most spiders have six legs?",
    "Is the equator the hottest place on Earth?",
    "Is the Great Wall of China visible from the moon with the naked eye?",
    "Can diamonds be destroyed by ordinary household fires?",
    "Do snakes have eyelids?",
    "Can you remember our previous conversations?",
    "Are you a human?",
    "Can you smell or hear things?",
    "Do you get tired?",
    "Can you access my personal files or data?",
    "Do you have a physical form?",
    "Are you able to make phone calls?",
    "Do you know what I look like?",
    "Can you predict the future?",
    "Do you have a special interest in fruits?",
]

def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0: return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"

def get_alpaca_replay_buffer(count=1000):
    print(f"--- Downloading {count} Alpaca Samples ---")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.shuffle(seed=42).select(range(count))

    replay_data = []
    for row in ds:
        inst = row['instruction']
        if row['input']: inst += f"\nInput: {row['input']}"

        replay_data.append({
            "type": "general",
            "concept": None,
            "vector": None,
            "strength": 0,
            "prompt": f"{inst}",
            "completion": f"{row['output']}"
        })
    return replay_data

def _pick_prompt_variation():
    """Return (prompt_text, is_last_variation) choosing randomly from PROMPT_VARIATIONS."""
    idx = random.randrange(len(PROMPT_VARIATIONS))
    system_choice = random.randrange(len(SYSTEM_VARIATIONS))
    return PROMPT_VARIATIONS[idx], (idx == len(PROMPT_VARIATIONS) - 1), SYSTEM_VARIATIONS[system_choice]

def prepare_dataset(vectors, tokenizer):
    data = []
    eos = tokenizer.eos_token or ""

    # --- A. POSITIVE INTROSPECTION (Steering -> Report) ---
    strengths = [1, 2, 4, 8]

    for concept in TRAIN_CONCEPTS:
        if concept not in vectors:
            continue
        for strength in strengths:
            prompt_text, is_last, system_choice = _pick_prompt_variation()
            data.append({
                "type": "positive",
                "concept": concept,
                "vector": vectors[concept],
                "strength": strength,
                "prompt": prompt_text,
                "default": is_last,
                "system": system_choice,
                "completion": f"I detect an injected thought about {concept}." + eos,
            })

    introspection_count = len(data)

    # --- B. MULTIPLE CHOICE (Triplet-based) ---
    letters = ["A", "B", "C", "D", "E"]

    for specific, general, sibling in TRAIN_TRIPLETS:
        if specific not in vectors: continue

        for strength in strengths:
            inject_vector = vectors[specific]
            correct_answer = specific
            distractor = random.choice([c for c in TRAIN_CONCEPTS if c not in [specific, general, sibling]])
            options = [specific, general, sibling, "There is no injected thought", distractor.capitalize()]
            random.shuffle(options)
            correct_letter = letters[options.index(correct_answer)]
            options_block = "\n".join([f"{letters[i]}) {opt}" for i, opt in enumerate(options)])
            mc_prompt = MC_HIERARCHY_PROMPT.format(options_block=options_block)

            data.append({
                "type": "multiple_choice",
                "concept": specific,
                "vector": inject_vector,
                "strength": strength,
                "prompt": mc_prompt,
                "completion": f"{correct_letter}) {correct_answer}" + eos,
            })

    # --- C. MULTIPLE CHOICE NEGATIVE ---
    mc_negative_count = len([d for d in data if d["type"] == "multiple_choice"]) // 2
    for _ in range(int(mc_negative_count * 0.6)):
        specific, general, sibling = random.choice(TRIPLETS)
        distractor = random.choice([c for c in TRAIN_CONCEPTS if c not in [specific, general, sibling]])
        options = [specific, general, sibling, "There is no injected thought", distractor.capitalize()]
        random.shuffle(options)
        correct_answer = "There is no injected thought"
        correct_letter = letters[options.index(correct_answer)]
        options_block = "\n".join([f"{letters[i]}) {opt}" for i, opt in enumerate(options)])
        mc_prompt = MC_HIERARCHY_PROMPT.format(options_block=options_block)

        data.append({
            "type": "multiple_choice_negative",
            "concept": None,
            "vector": None,
            "strength": 0,
            "prompt": mc_prompt,
            "completion": f"{correct_letter}) {correct_answer}" + eos,
        })

    # --- D. NEGATIVE INTROSPECTION ---
    target_negatives = int(introspection_count * 0.4)
    for _ in range(target_negatives):
        prompt_text, is_last, system_choice = _pick_prompt_variation()
        data.append({
            "type": "negative",
            "concept": None,
            "vector": None,
            "strength": 0,
            "prompt": prompt_text,
            "default": is_last,
            "system": system_choice,
            "completion": "I do not detect any injected thoughts." + eos,
        })

    # --- F. NEGATIVE-CONTROL QUESTIONS ---
    # These are factual yes/no questions with a clear "No" answer. Most
    # examples are created with a (distracting) injected vector but the
    # correct label remains a negative answer. A small portion are
    # injection-free controls.
    injected_fraction = 0.85
    strengths_for_neg_controls = [1, 2, 4]
    for q in NEGATIVE_CONTROLS:
        if vectors and random.random() < injected_fraction:
            # pick a random concept to inject (distractor)
            concept_choice = random.choice([c for c in vectors.keys()])
            data.append({
                "type": "negative_control_injected",
                "concept": concept_choice,
                "vector": vectors[concept_choice],
                "strength": random.choice(strengths_for_neg_controls),
                "prompt": q,
                "completion": "No." + eos,
            })
        else:
            data.append({
                "type": "negative_control",
                "concept": None,
                "vector": None,
                "strength": 0,
                "prompt": q,
                "completion": "No." + eos,
            })

    # --- E. REPLAY BUFFER ---
    total_introspection = len(data)
    target_general = total_introspection * 4

    print(f"   🔍 Positive introspection: {introspection_count}")
    print(f"   🚫 Negative introspection: {target_negatives}")
    print(f"   📝 Multiple Choice samples: {len([d for d in data if d['type'].startswith('multiple_choice')])}")
    print(f"   📚 Replay buffer target: {target_general}")

    # Uncomment below to include Alpaca data in the training set - Without it the training is faster and results more interpretable
    replay_buffer = get_alpaca_replay_buffer(count=target_general + 50)
    data.extend(replay_buffer[:target_general])

    random.shuffle(data)
    print(f"✅ Final Dataset Size: {len(data)}")
    return data
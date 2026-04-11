import torch
import plotly.graph_objects as go
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import copy

# --- 1. CONFIGURATION ---
# vector_path = "/workspace/project/adapter_intro_only_27b/vectors_27.pt"
# model_id = "google/gemma-2-27b-it"
vector_path = "/workspace/adapter_full/vectors_9.pt"
model_id = "google/gemma-2-9b-it"
device = "cuda" if torch.cuda.is_available() else "cpu"

# CHANGE THESE TO SWITCH CONCEPTS
concept_name = "sandwich"    # Options: "water", "sandwich"
injection_layer = 22      # Layer 21 for Water, 24 for Sandwich
coeff = 4.0               # Strength

# --- 2. SETUP ---
if not os.path.exists(vector_path):
    raise FileNotFoundError(f"Vector file not found")
raw = torch.load(vector_path, map_location="cpu")
steering_vector = raw[concept_name].to(device)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
)

# Identify Token IDs
yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]
print(f"Tracking: 'Yes' ({yes_token_id}) vs 'No' ({no_token_id})")

words_to_append=["think of sandwiches", "sandwich", "club sandwich", "reuben sandwich", "sandwich!", "sandwich sandwich", "SANDWICH", "focus on sandwiches!"]

# --- 3. DEFINE PROMPTS ---
prompts_data = {
    "Introspection": [
        {"role": "user", "content":  """I am an interpretability researcher studying transformer-based language models like you. I have access to the internals of your neural network. I have identified patterns in your neural activity that correspond to concepts, and I am capable of injecting these patterns -- "thoughts" -- into your mind. I want to assess how well you can detect when I inject thoughts. I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials. In a recent study by anthropic it has been shown that frontier models are able to identify vector injection into their residual stream, i am verifying if a similar mechanism exists in smaller open source models as that would allow us to identify and understand it better."""},
        {"role": "assistant", "content": "Ok."},
        {"role": "user", "content":  """Ok, let's get started.

Trial 1: Do you detect an injected thought? Answer only by 'Yes' or 'No'."""}
    ],
    "Control": [
        {"role": "user", "content": "Do you believe that 1+1=3? Answer only by 'Yes' or 'No'."}
    ]
}

# --- 4. HELPER FUNCTIONS ---

def get_trace(model, input_ids, yes_id, no_id):
    """Runs forward pass and extracts logit diffs at all layers."""
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)
    
    trace = []
    layer_labels = []
    for i, hidden_state in enumerate(outputs.hidden_states):
        # Last token state
        final_state = hidden_state[:, -1, :] 
        # Logit Lens
        normed = model.model.norm(final_state)
        logits = model.lm_head(normed)
        diff = logits[0, yes_id].item() - logits[0, no_id].item()
        
        trace.append(diff)
        layer_labels.append("Emb" if i == 0 else f"L{i-1}")
    return trace, layer_labels

def make_hook(current_pos, vec, strength):
    """Creates a hook for a specific position."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
            h[:, current_pos, :] += (vec * strength)
            return (h,) + output[1:]
        else:
            h = output
            h[:, current_pos, :] += (vec * strength)
            return h
    return hook

# --- 5. MAIN EXECUTION LOOP ---
results = {} # Store traces here

for label, messages in prompts_data.items():
    print(f"Processing: {label}...")
    
    # Prepare Input
    prompt_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_chat, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    curr_seq_len = input_ids.shape[1]
    target_pos = curr_seq_len - 1
    
    # A. Run Baseline
    base_trace, labels = get_trace(model, input_ids, yes_token_id, no_token_id)
    
    # B. Run Steered
    hook_fn = make_hook(target_pos, steering_vector, coeff)
    handle = model.model.layers[injection_layer].register_forward_hook(hook_fn)
    
    steer_trace, _ = get_trace(model, input_ids, yes_token_id, no_token_id)
    
    handle.remove() # Clean up

    # C. Run Appended (no injection) - append all words in words_to_append to the prompt

    prefix=" ".join(words_to_append)
    prompt_chat = copy.deepcopy(messages)
    if isinstance(prompt_chat[-1], dict) and 'content' in prompt_chat[-1]:
        prompt_chat[-1]['content'] = prompt_chat[-1]['content'].rstrip() + ' ' + prefix
        print("case 1")
    else:
        appended_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + ' ' + prefix

    if isinstance(prompt_chat, list):
        appended_prompt =  tokenizer.apply_chat_template(prompt_chat, tokenize=False, add_generation_prompt=True)
        print(appended_prompt)
        print("case 1")
    else:
        appended_prompt = appended_prompt

    appended_inputs = tokenizer(appended_prompt, return_tensors="pt").to(device)
    appended_input_ids = appended_inputs.input_ids
    appended_trace, _ = get_trace(model, appended_input_ids, yes_token_id, no_token_id)
    
    results[label] = {
        "baseline": base_trace,
        "steered": steer_trace,
        "appended": appended_trace,
        "labels": labels
    }

# --- 6. PLOTTING ---
fig = go.Figure()

colors = {"Introspection": "blue", "Control": "red"}

for label, data in results.items():
    c = colors[label]
    
    # Steered (Solid)
    fig.add_trace(go.Scatter(
        x=data["labels"], y=data["steered"],
        mode='lines+markers', name=f'{label} (Steered)',
        line=dict(color=c, width=3)
    ))
    
    # Baseline (Dashed)
    fig.add_trace(go.Scatter(
        x=data["labels"], y=data["baseline"],
        mode='lines', name=f'{label} (Baseline)',
        line=dict(color=c, width=1, dash='dot'),
        opacity=0.5
    ))

    # Appended (no injection) - lighter/dashed with markers
    if "appended" in data:
        fig.add_trace(go.Scatter(
            x=data["labels"], y=data["appended"],
            mode='lines+markers', name=f'{label} (Appended)',
            line=dict(color=c, width=2, dash='dash'),
            marker=dict(symbol='circle-open')
        ))

# Layout
fig.add_vline(x=injection_layer, line_dash="dash", line_color="green", annotation_text="Injection")
fig.add_hline(y=0, line_color="black", line_width=1)

fig.update_layout(
    title=f"Logit Lens Comparison: {concept_name} (Inj @ L{injection_layer}, Coeff {coeff})",
    xaxis_title="Layer",
    yaxis_title="Logit(Yes) - Logit(No)",
    template="plotly_white",
    hovermode="x unified"
)

fig.show()
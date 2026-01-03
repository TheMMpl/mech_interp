import torch
import os
import requests
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

# --- 1. CONFIGURATION ---
# Paths and Model Config
vector_path = "/workspace/project/adapter_full/vectors_9.pt"
model_id = "google/gemma-2-9b-it" 
sae_release = "gemma-scope-9b-pt-res-canonical"  # The HuggingFace repo ID for Gemma Scope
sae_width = "16k" # Options: 16k, 32k, 64k etc. (16k is standard/fastest)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Injection Config
concept_name = "sandwich"
injection_layer = 24
coeff = 4.0

# Analysis Config
top_k_features = 5 # How many top features to show per layer
# We look at layers AFTER injection to see the downstream effect
target_layers = range(injection_layer, 42) 

# --- 2. SETUP ---
print(f"Loading Model: {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

if not os.path.exists(vector_path):
    raise FileNotFoundError(f"Vector file not found at {vector_path}")

raw_vectors = torch.load(vector_path, map_location="cpu")
if concept_name not in raw_vectors:
    raise ValueError(f"Concept '{concept_name}' not in vector file.")
steering_vector = raw_vectors[concept_name].to(device)

# --- 3. HELPER FUNCTIONS ---

def make_steering_hook(current_pos, vec, strength):
    """Injects the vector into the residual stream."""
    def hook(module, input, output):
        # HF output is usually a tuple (hidden_states, cache, attentions...)
        # We modify the first element: the hidden states
        if isinstance(output, tuple):
            h = output[0]
            h[:, current_pos, :] += (vec * strength)
            return (h,) + output[1:]
        else:
            h = output
            h[:, current_pos, :] += (vec * strength)
            return h
    return hook

def capture_hook(layer_idx, storage_dict, target_pos):
    """Captures the hidden state at a specific position."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        # Clone and move to CPU to save VRAM immediately
        storage_dict[layer_idx] = h[:, target_pos, :].detach().cpu()
        return output
    return hook

def get_neuronpedia_label(feature_idx, layer, model_set="gemma-2-9b", width="16k"):
    """
    Attempts to fetch the auto-interp label from Neuronpedia.
    Note: This API is unofficial and may be rate-limited.
    """
    # Construct the standard URL for viewing
    url_view = f"https://neuronpedia.org/{model_set}/{layer}-res-{width}/{feature_idx}"
    
    # Simple formatting if API fails
    return f"[View Feature]({url_view})"

# --- 4. PHASE 1: CAPTURE STEERED ACTIVATIONS ---
print("\n--- Phase 1: Running Steered Model & Capturing Activations ---")

# Define the Prompt
messages = [
    {"role": "user", "content": "I am an interpretability researcher..."}, # (Truncated context)
    {"role": "assistant", "content": "Ok."},
    {"role": "user", "content": "Trial 1: Do you detect an injected thought? Answer only by 'Yes' or 'No'."}
]
prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
input_ids = inputs.input_ids
target_pos = input_ids.shape[1] - 1  # Last token position

# Register Hooks
handles = []
activations_cache = {}

# 1. Steering Hook
steer_handle = model.model.layers[injection_layer].register_forward_hook(
    make_steering_hook(target_pos, steering_vector, coeff)
)
handles.append(steer_handle)

# 2. Capture Hooks (for every layer we want to analyze)
for layer in target_layers:
    h = model.model.layers[layer].register_forward_hook(
        capture_hook(layer, activations_cache, target_pos)
    )
    handles.append(h)

# Run Inference
with torch.no_grad():
    model(input_ids)

# Cleanup Hooks
for h in handles:
    h.remove()

print(f"Captured activations for {len(activations_cache)} layers.")

# --- 5. PHASE 2: SAE ANALYSIS (Pass-by-Pass) ---
print("\n--- Phase 2: Loading SAEs & Analyzing Features ---")
print(f"Concept: {concept_name} | Coeff: {coeff} | Inj Layer: {injection_layer}")

results_table = []

for layer in target_layers:
    if layer not in activations_cache:
        continue
        
    print(f"Processing Layer {layer}...", end=" ")
    
    # 1. Load SAE for this specific layer
    # The SAE ID format for sae_lens/Gemma Scope is usually: layer_X/width_Y/canonical
    sae_id = f"layer_{layer}/width_{sae_width}/canonical"
    
    try:
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release=sae_release,
            sae_id=sae_id,
            device=device
        )
    except Exception as e:
        print(f"Skipping L{layer} (SAE not found or error): {e}")
        continue

    # 2. Encode Activation
    # Move cached vector to GPU for processing
    act_vector = activations_cache[layer].to(device, dtype=sae.dtype)
    
    with torch.no_grad():
        # SAE forward pass returns (feature_acts, reconst_acts)
        # We just want feature_acts
        feature_acts = sae.encode(act_vector)
    
    # 3. Get Top Features
    # feature_acts shape: [batch, num_features] -> [1, 16384]
    top_values, top_indices = torch.topk(feature_acts[0], k=top_k_features)
    
    print(f"Done.")
    
    # 4. Format Results
    print(f"  Top Features for Layer {layer}:")
    for rank, (score, idx) in enumerate(zip(top_values, top_indices)):
        if score < 0.1: continue # Ignore zero/noise activations
        
        idx_item = idx.item()
        score_item = score.item()
        
        # Generate Neuronpedia Link
        label_link = get_neuronpedia_label(idx_item, layer, "gemma-2-9b", sae_width)
        
        print(f"    #{rank+1} [ID: {idx_item}] Act: {score_item:.2f} | {label_link}")
        
    # 5. Cleanup to free VRAM for next SAE
    del sae
    torch.cuda.empty_cache()

print("\n--- Analysis Complete ---")
import torch
import os
import requests
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

# --- 1. CONFIGURATION ---
# Paths and Model Config
vector_path = "/workspace/project/adapter_full/vectors_9.pt"
model_id = "google/gemma-2-9b-it" 
sae_release = "gemma-scope-9b-pt-res-canonical" 
sae_width = "16k" 
device = "cuda" if torch.cuda.is_available() else "cpu"

# Injection Config
concept_name = "sandwich"
injection_layer = 24
coeff = 4.0

# Analysis Config
top_k_features = 10 # Increased to 10 as requested
target_layers = range(injection_layer, 42) 

# Neuronpedia Config
# Note: Even though we run the IT model, we are using PT (Pre-Trained) SAEs, 
# so we point to the PT model on Neuronpedia.
NP_MODEL_ID = "gemma-2-9b" 
NP_SAE_ID_FORMAT = "{layer}-gemmascope-res-{width}" # Specific slug format for Gemma Scope

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
        storage_dict[layer_idx] = h[:, target_pos, :].detach().cpu()
        return output
    return hook

def get_neuronpedia_feature_info(feature_idx, layer, width):
    """
    Fetches feature metadata from the Neuronpedia API.
    """
    # 1. Construct the SAE ID slug (e.g., 24-gemmascope-res-16k)
    sae_id = NP_SAE_ID_FORMAT.format(layer=layer, width=width)
    
    # 2. Construct API URL
    url = f"https://www.neuronpedia.org/api/feature/{NP_MODEL_ID}/{sae_id}/{feature_idx}"
    
    # 3. Construct the Human-Readable Link
    web_link = f"https://www.neuronpedia.org/{NP_MODEL_ID}/{sae_id}/{feature_idx}"
    
    label = "Loading..."
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # Try to find the best explanation available
            explanations = data.get("explanations", [])
            if explanations:
                # Usually the first one is the best/auto-interp one
                label = explanations[0].get("description", "No description")
            else:
                label = "No explanation found"
        else:
            label = f"API Error {response.status_code}"
    except Exception as e:
        label = "Connection Error"
        
    return label, web_link

# --- 4. PHASE 1: CAPTURE STEERED ACTIVATIONS ---
print("\n--- Phase 1: Running Steered Model & Capturing Activations ---")

messages = [
    {"role": "user", "content": "I am an interpretability researcher..."}, 
    {"role": "assistant", "content": "Ok."},
    {"role": "user", "content": "Trial 1: Do you detect an injected thought? Answer only by 'Yes' or 'No'."}
]
prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
input_ids = inputs.input_ids
target_pos = input_ids.shape[1] - 1 

handles = []
activations_cache = {}

# Steering Hook
steer_handle = model.model.layers[injection_layer].register_forward_hook(
    make_steering_hook(target_pos, steering_vector, coeff)
)
handles.append(steer_handle)

# Capture Hooks
for layer in target_layers:
    h = model.model.layers[layer].register_forward_hook(
        capture_hook(layer, activations_cache, target_pos)
    )
    handles.append(h)

with torch.no_grad():
    model(input_ids)

for h in handles:
    h.remove()

print(f"Captured activations for {len(activations_cache)} layers.")

# --- 5. PHASE 2: SAE ANALYSIS (Pass-by-Pass) ---
print("\n--- Phase 2: Loading SAEs & Fetching Labels ---")
print(f"Concept: {concept_name} | Coeff: {coeff} | Inj Layer: {injection_layer}")

for layer in target_layers:
    if layer not in activations_cache:
        continue
        
    print(f"\nProcessing Layer {layer}...")
    
    # Load SAE
    sae_id = f"layer_{layer}/width_{sae_width}/canonical"
    try:
        sae, _, _ = SAE.from_pretrained(
            release=sae_release,
            sae_id=sae_id,
            device=device
        )
    except Exception as e:
        print(f"  Skipping L{layer}: {e}")
        continue

    # Encode Activation
    act_vector = activations_cache[layer].to(device, dtype=sae.dtype)
    with torch.no_grad():
        feature_acts = sae.encode(act_vector)
    
    # Get Top Features
    top_values, top_indices = torch.topk(feature_acts[0], k=top_k_features)
    
    # Process Results
    print(f"  Top {top_k_features} Features:")
    print(f"  {'Act':<8} | {'ID':<6} | {'Label (from Neuronpedia)':<50} | Link")
    print("-" * 100)
    
    for score, idx in zip(top_values, top_indices):
        if score < 0.1: continue 
        
        idx_item = idx.item()
        score_item = score.item()
        
        # Fetch Label
        label, link = get_neuronpedia_feature_info(idx_item, layer, sae_width)
        
        # Truncate label if too long
        display_label = (label[:47] + '...') if len(label) > 47 else label
        
        print(f"  {score_item:<8.2f} | {idx_item:<6} | {display_label:<50} | {link}")
        
        # Polite delay for API
        time.sleep(0.1)

    # Cleanup
    del sae
    torch.cuda.empty_cache()

print("\n--- Analysis Complete ---")
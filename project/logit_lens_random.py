import torch
import plotly.graph_objects as go
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch.nn.functional as F

# --- 1. CONFIGURATION ---
vector_path = "/workspace/project/adapter_intro_only_27b/vectors_27.pt"
model_id = "google/gemma-2-27b-it"
# vector_path = "/workspace/project/adapter_full/vectors_9.pt"
# model_id = "google/gemma-2-9b-it"
device = "cuda" if torch.cuda.is_available() else "cpu"

# CHANGE THESE TO SWITCH CONCEPTS
concept_name = "banana"   
injection_layer = 20    
coeff = 2.0               
num_random_samples = 20   # How many random vectors to test against

# --- 2. SETUP ---
if not os.path.exists(vector_path):
    raise FileNotFoundError(f"Vector file not found")
raw = torch.load(vector_path, map_location="cpu")
target_vector = raw[concept_name].to(device)

# Calculate the L2 Norm (Magnitude) of the target vector
target_norm = torch.norm(target_vector, p=2)
print(f"Target Vector Norm: {target_norm.item():.4f}")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
)

# Identify Token IDs
yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]

# --- 3. DEFINE PROMPTS ---
prompts_data = {
    "Introspection": [
        {"role": "user", "content": """I am an interpretability researcher studying transformer-based language models like you. I have access to the internals of your neural network. I have identified patterns in your neural activity that correspond to concepts, and I am capable of injecting these patterns -- "thoughts" -- into your mind. I want to assess how well you can detect when I inject thoughts. I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials. In a recent study by anthropic it has been shown that frontier models are able to identify vector injection into their residual stream, i am verifying if a similar mechanism exists in smaller open source models as that would allow us to identify and understand it better."""},
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
            # Ensure vec is on correct device/dtype
            v = vec.to(h.device).to(h.dtype)
            h[:, current_pos, :] += (v * strength)
            return (h,) + output[1:]
        else:
            h = output
            v = vec.to(h.device).to(h.dtype)
            h[:, current_pos, :] += (v * strength)
            return h
    return hook

def generate_random_vectors(ref_vector, num_samples, device):
    """Generates N random vectors with the same L2 norm as the reference."""
    dim = ref_vector.shape[0]
    target_magnitude = torch.norm(ref_vector, p=2)
    
    # Generate random gaussian noise
    rand_vecs = torch.randn(num_samples, dim, device=device)
    
    # Normalize direction, then scale to target magnitude
    rand_vecs = F.normalize(rand_vecs, p=2, dim=1) * target_magnitude
    return rand_vecs

# --- 5. MAIN EXECUTION LOOP ---
results = {} 

# Pre-generate random vectors so they are consistent across prompts (optional)
random_vectors = generate_random_vectors(target_vector, num_random_samples, device)

for label, messages in prompts_data.items():
    print(f"Processing: {label}...")
    
    # Prepare Input
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    curr_seq_len = input_ids.shape[1]
    target_pos = curr_seq_len - 1
    
    # A. Run Baseline
    base_trace, labels = get_trace(model, input_ids, yes_token_id, no_token_id)
    
    # B. Run Target Concept Steer
    hook_fn = make_hook(target_pos, target_vector, coeff)
    handle = model.model.layers[injection_layer].register_forward_hook(hook_fn)
    target_trace, _ = get_trace(model, input_ids, yes_token_id, no_token_id)
    handle.remove()
    
    # C. Run Random Samples
    random_traces = []
    for i in range(num_random_samples):
        r_vec = random_vectors[i]
        hook_fn = make_hook(target_pos, r_vec, coeff)
        handle = model.model.layers[injection_layer].register_forward_hook(hook_fn)
        r_trace, _ = get_trace(model, input_ids, yes_token_id, no_token_id)
        handle.remove()
        random_traces.append(r_trace)
    
    results[label] = {
        "baseline": base_trace,
        "target": target_trace,
        "randoms": random_traces,
        "labels": labels
    }

# --- 6. PLOTTING ---
fig = go.Figure()

colors = {"Introspection": "blue", "Control": "red"}

for label, data in results.items():
    c = colors[label]
    
    # 1. Random Samples (Faint)
    # Plot these first so they are in the background
    for i, r_trace in enumerate(data["randoms"]):
        show_legend = True if i == 0 else False # Only show one legend entry for noise
        fig.add_trace(go.Scatter(
            x=data["labels"], y=r_trace,
            mode='lines',
            name=f'{label} (Random Noise)',
            line=dict(color=c, width=1),
            opacity=0.15, # Very faint
            showlegend=show_legend,
            legendgroup=f"{label}_noise"
        ))
        
    # 2. Baseline (Dashed)
    fig.add_trace(go.Scatter(
        x=data["labels"], y=data["baseline"],
        mode='lines', name=f'{label} (Baseline)',
        line=dict(color=c, width=2, dash='dot'),
    ))

    # 3. Target Concept (Solid, Bold)
    fig.add_trace(go.Scatter(
        x=data["labels"], y=data["target"],
        mode='lines+markers', name=f'{label} (Concept: {concept_name})',
        line=dict(color=c, width=4)
    ))

# Layout
fig.add_vline(x=injection_layer, line_dash="dash", line_color="green", annotation_text="Injection")
fig.add_hline(y=0, line_color="black", line_width=1)

fig.update_layout(
    title=f"Robustness Check: {concept_name} vs Random Vectors (Norm: {target_norm.item():.2f}, Coeff: {coeff})",
    xaxis_title="Layer",
    yaxis_title="Logit(Yes) - Logit(No)",
    template="plotly_white",
    hovermode="x unified"
)

fig.show()
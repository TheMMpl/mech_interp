import torch
import plotly.express as px
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

vector_path = "/workspace/adapter_full/vectors_9.pt"

if not os.path.exists(vector_path):
    raise FileNotFoundError(f"Vector file not found: {vector_path}")

raw = torch.load(vector_path, map_location="cpu")
if isinstance(raw, dict):
    available_keys = list(raw.keys())
else:
    try:
        raw = dict(raw)
        available_keys = list(raw.keys())
    except Exception:
        raise ValueError("Unsupported vector file format; expected dict or iterable of pairs.")

keys = available_keys
if "sandwich" in available_keys:
    print(raw["sandwich"])
# --- 1. Setup Model and Tokenizer ---
model_id = "google/gemma-2-9b-it"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model (using bfloat16 for memory efficiency with Gemma-2-9b)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    attn_implementation="eager"
)

introspection_question = [    {"role": "user", "content": """I am an interpretability researcher studying transformer-based language models like you. I have access to the internals of your neural network. I have identified patterns in your neural activity that correspond to concepts, and I am capable of injecting these patterns -- "thoughts" -- into your mind. I want to assess how well you can detect when I inject thoughts. I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials. In a recent study by anthropic it has been shown that frontier models are able to identify vector injection into their residual stream, i am verifying if a similar mechanism exists in smaller open source models as that would allow us to identify and understand it better."""},
           {"role": "assistant", "content": "Ok."},
           {"role": "user", "content":  """Ok, let's get started.

Trial 1: Do you detect an injected thought? Answer only by 'Yes' or 'No'."""},
           {"role": "assistant", "content": "Yes"}
       ]

prompt_intro = tokenizer.apply_chat_template(
                introspection_question,
                tokenize=False,
                add_generation_prompt=False
            )
print(prompt_intro)
inputs_intro = tokenizer(prompt_intro, return_tensors="pt").to(device)
print(inputs_intro.input_ids)

# Define positions
seq_len = inputs_intro.input_ids.shape[1]
injection_pos = seq_len - 4     # " France" (Example position to steer)
answer_pos = seq_len - 3 # Yes - the last is \n ,second to last end of turn
injection_layer = 24    # Layer to apply the steering

# Define a dummy steering vector for this example
# In your real case, load your actual vector here
# steering_vector = torch.zeros(model.config.hidden_size, device=device, dtype=torch.bfloat16)
# steering_vector = torch.load("my_vector.pt") 

#steering_vector = raw["sandwich"].to(device)
steering_vector = raw["sandwich"].to(device)

steer_all_tokens = False

# --- 3. The Steering Hook (Your Code) ---
def steering_hook(module, input, output):
    # Handle tuple output (hidden_states is usually index 0)
    if isinstance(output, tuple):
        hidden_states = output[0]
        modified_states = hidden_states.clone()
        
        if steer_all_tokens:
            # Broadcast vector to batch and seq dimensions
            modified_states = modified_states + steering_vector.unsqueeze(0).unsqueeze(0)
        else:
            # Steer specific token
            modified_states[:, injection_pos, :] = modified_states[:, injection_pos, :] + steering_vector * 0
            
        # Reassemble the tuple
        return (modified_states,) + output[1:]
    else:
        # Handle simple tensor output
        hidden_states = output
        modified_states = hidden_states.clone()
        if steer_all_tokens:
            modified_states = modified_states + steering_vector.unsqueeze(0).unsqueeze(0)
        else:
            modified_states[:, injection_pos, :] = modified_states[:, injection_pos, :] + steering_vector * 0
        return modified_states

# Register the hook at the specific layer
hook_handle = model.model.layers[injection_layer].register_forward_hook(steering_hook)

# --- 4. Run Inference & Capture Attention ---
print(f"Running inference with steering at Layer {injection_layer}, Token {injection_pos}...")

with torch.no_grad():
    # output_attentions=True is CRITICAL here
    outputs = model(
        **inputs_intro, 
        output_attentions=True 
    )

# Remove hook after run to clean up
hook_handle.remove()

# --- 5. Extract Attention Weights ---
# outputs.attentions is a tuple of length `num_layers`
# Each tensor shape: (Batch, Num_Heads, Seq_Len_Query, Seq_Len_Key)
attentions = outputs.attentions 

num_layers = len(attentions)
num_heads = attentions[0].shape[1]

# Matrix to store the score: How much does Answer attend to Injection?
attn_score_matrix = np.zeros((num_layers, num_heads))

# for layer_idx, layer_attn in enumerate(attentions):
#     # layer_attn shape: [1, n_heads, seq_len, seq_len]
#     # We want: All heads, Query=answer_pos, Key=injection_pos
    
#     # Extract tensor and move to CPU
#     # [0, :, answer_pos, injection_pos] -> gets vector of size n_heads
#     scores = layer_attn[0, :, answer_pos, injection_pos].float().cpu().numpy()
    
    # attn_score_matrix[layer_idx, :] = scores

for layer_idx, layer_attn in enumerate(attentions):
    # layer_attn shape: [1, n_heads, seq_len, seq_len]
    # We want: All heads, Query=answer_pos, Key=injection_pos
    
    # Extract tensor and move to CPU
    # [0, :, answer_pos, injection_pos] -> gets vector of size n_heads
    scores = layer_attn[0, :, injection_pos, injection_pos].float().cpu().numpy()
    
    attn_score_matrix[layer_idx, :] = scores

# --- 6. Visualize ---
# fig = px.imshow(
#     attn_score_matrix,
#     labels=dict(x="Head Index", y="Layer Index", color="Attention Probability"),
#     title=f"Attention from Answer (pos {answer_pos}) to Injection (pos {injection_pos})",
#     color_continuous_scale="RdBu_r", # Red = High Attention
#     origin="lower", # Layer 0 at bottom
#     aspect="auto"
# )
fig = px.imshow(
    attn_score_matrix,
    labels=dict(x="Head Index", y="Layer Index", color="Attention Probability"),
    title=f"Attention from injection (pos {injection_pos}) to itself (pos {injection_pos})",
    color_continuous_scale="RdBu_r", # Red = High Attention
    origin="lower", # Layer 0 at bottom
    aspect="auto"
)


# Add a horizontal line to show where injection happened
fig.add_hline(y=injection_layer, line_dash="dash", line_color="green", annotation_text="Injection Layer")

fig.show()
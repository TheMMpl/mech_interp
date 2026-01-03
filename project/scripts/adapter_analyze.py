import torch

def analyze_attention_sae_lora(model, sae, layer_idx):
    """
    Analyzes LoRA using an SAE trained on the 'z' vector (Concatenated Head Outputs).
    """
    # 1. Get Layer and Dimensions
    layer = model.base_model.model.model.layers[layer_idx]
    self_attn = layer.self_attn
    
    # Gemma 2 specifics
    num_heads = model.config.num_attention_heads
    head_dim = model.config.head_dim
    # The SAE input dimension should be num_heads * head_dim
    
    results = {}

    # --- Helper: Get LoRA Delta ---
    def get_lora_delta(module):
        if not hasattr(module, "lora_A"): return None
        A = module.lora_A["default"].weight
        B = module.lora_B["default"].weight
        scale = module.scaling["default"]
        return (B @ A) * scale

    # ==================================================
    # ANALYSIS 1: The Value (V) LoRA
    # "What raw messages is the LoRA inserting into the heads?"
    # ==================================================
    delta_V = get_lora_delta(self_attn.v_proj)
    # Shape: (num_kv_heads * head_dim, d_model)
    
    if delta_V is not None:
        # Note: Gemma 2 uses GQA (Grouped Query Attention).
        # The SAE expects (num_heads * head_dim), but delta_V corresponds to (num_kv_heads * head_dim).
        # We must repeat the delta_V rows to match the full 'z' vector space.
        
        num_kv_heads = model.config.num_key_value_heads
        num_groups = num_heads // num_kv_heads
        
        # Reshape to (num_kv_heads, head_dim, d_model)
        delta_V_reshaped = delta_V.reshape(num_kv_heads, head_dim, -1)
        
        # Repeat for each group to simulate the full 'z' vector construction
        # Shape: (num_kv_heads, num_groups, head_dim, d_model) -> (num_heads, head_dim, d_model)
        delta_V_expanded = delta_V_reshaped.unsqueeze(1).expand(-1, num_groups, -1, -1)
        delta_V_full = delta_V_expanded.reshape(num_heads * head_dim, -1)
        
        # Now we project this "Virtual Z Update" onto the SAE
        # We want to know: "If this LoRA activates, what Attn SAE features light up?"
        # SAE Decoder: (n_features, z_dim)
        # Delta V Full: (z_dim, d_model_in)
        
        # Project columns (input directions) or rows (output directions)?
        # V writes TO 'z'. So we project the COLUMNS of delta_V (the output side).
        # We sum over the input dimension (d_model) to get the "general vibe".
        
        # Normalize
        sae_dec_norm = torch.nn.functional.normalize(sae.W_dec, dim=1)
        delta_v_norm = torch.nn.functional.normalize(delta_V_full, dim=0)
        
        projection = sae_dec_norm @ delta_v_norm # (n_features, d_model)
        scores = projection.sum(dim=1)
        
        top_scores, top_feats = torch.topk(scores, k=5)
        results['v_write_features'] = top_feats.tolist()

    # ==================================================
    # ANALYSIS 2: The Output (O) LoRA
    # "How is the LoRA changing how head messages are read?"
    # ==================================================
    delta_O = get_lora_delta(self_attn.o_proj)
    # Shape: (d_model, num_heads * head_dim)
    
    if delta_O is not None:
        # O reads FROM 'z'. So we project the ROWS of delta_O (the input side).
        # This tells us: "Which SAE features (head messages) does this LoRA listen to more?"
        
        delta_o_input_dirs = delta_O.T # (z_dim, d_model)
        
        # Normalize
        delta_o_norm = torch.nn.functional.normalize(delta_o_input_dirs, dim=0)
        sae_dec_norm = torch.nn.functional.normalize(sae.W_dec, dim=1)

        # Dot product
        projection = sae_dec_norm @ delta_o_norm
        scores = projection.sum(dim=1)
        
        top_scores, top_feats = torch.topk(scores, k=5)
        results['o_read_features'] = top_feats.tolist()

    return results

import requests
import time

def get_neuronpedia_label(model_id, sae_id, feature_index):
    """
    Fetches the auto-interpretability label for a specific SAE feature.
    """
    url = f"https://www.neuronpedia.org/api/feature/{model_id}/{sae_id}/{feature_index}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # The structure often has explanations in 'explanations' list
            # We grab the description from the first explanation if available
            explanations = data.get("explanations", [])
            if explanations:
                return explanations[0].get("description", "No description found")
            return "No explanation available"
        else:
            return f"Error {response.status_code}"
    except Exception as e:
        return f"Request failed: {str(e)}"

# Example Usage with your SAE analysis
def print_readable_lora_analysis(analysis_results, model_slug, sae_slug):
    """
    Takes the output from the previous analyze_attention_sae_lora function
    and prints human-readable labels.
    """
    print(f"--- Semantic LoRA Analysis ({sae_slug}) ---")
    
    # Check V-Write Features (What the LoRA is writing)
    if 'v_write_features' in analysis_results:
        print("\n[+] LoRA is injecting these concepts (V-Write):")
        for feat_idx in analysis_results['v_write_features']:
            label = get_neuronpedia_label(model_slug, sae_slug, feat_idx)
            print(f"  - Feature {feat_idx}: {label}")
            # Rate limit to be nice to the API
            time.sleep(0.1) 

    # Check O-Read Features (What the LoRA is attending to)
    if 'o_read_features' in analysis_results:
        print("\n[+] LoRA is attending to these concepts (O-Read):")
        for feat_idx in analysis_results['o_read_features']:
            label = get_neuronpedia_label(model_slug, sae_slug, feat_idx)
            print(f"  - Feature {feat_idx}: {label}")
            time.sleep(0.1)

if __name__== "__main__":
    from transformers import AutoModelForCausalLM
    from sae_lens import SAE  # pip install sae-lens
    # Configuration
    MODEL_NAME = "google/gemma-2-27b-it"
    HF_TOKEN = os.getenv("HF_TOKEN")
    sae_id = "layer_0/width_16k/canonical"
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release = "gemma-scope-9b-pt-att-canonical",
        sae_id = "layer_0/width_16k/canonical",
    )


    # Load Model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN, device_map="auto", torch_dtype=torch.bfloat16)



    # Analyze LoRA at Layer 24
    layer_idx = 24
    analysis_results = analyze_attention_sae_lora(model, sae, layer_idx)

    # Print Human-Readable Analysis
    model_slug = MODEL_NAME.replace("/", "-")
    print_readable_lora_analysis(analysis_results, model_slug, sae_id)
import torch
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import copy
from typing import List, Optional

# --- Configuration ---
vector_path = "/workspace/adapter_full/vectors_9.pt"
model_id = "google/gemma-2-9b-it"
concept_key = "sandwich"
injection_layer = 24
steering_scale = 4
steer_all_tokens = False

# Set to a list of words to append after the answer token, e.g. ["water", "is", "refreshing"]
# Set to None to run the original single-plot analysis.
append_words: Optional[List[str]] = ["think of sandwiches", "sandwich", "club sandwich", "reuben sandwich", "sandwich!", "sandwich sandwich", "SANDWICH", "focus on sandwiches!"]


def load_vectors(path: str) -> dict:
    """Load steering vectors from file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vector file not found: {path}")
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict):
        return raw
    try:
        return dict(raw)
    except Exception:
        raise ValueError("Unsupported vector file format; expected dict or iterable of pairs.")


def build_prompt(tokenizer, messages, extra_words=None):
    """Build prompt text, optionally appending words to the last user message (before the answer)."""
    msgs = copy.deepcopy(messages)
    if extra_words:
        prefix = " ".join(extra_words)
        # Find the last user message and append there (before the assistant answer)
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i]["role"] == "user":
                msgs[i]["content"] = msgs[i]["content"].rstrip() + " " + prefix
                break
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


def make_steering_hook(steering_vector, injection_pos, scale, steer_all):
    """Return a forward-hook that adds ``steering_vector`` to the residual stream."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0].clone()
            if steer_all:
                h = h + steering_vector.unsqueeze(0).unsqueeze(0)
            else:
                h[:, injection_pos, :] = h[:, injection_pos, :] + steering_vector * scale
            return (h,) + output[1:]
        else:
            h = output.clone()
            if steer_all:
                h = h + steering_vector.unsqueeze(0).unsqueeze(0)
            else:
                h[:, injection_pos, :] = h[:, injection_pos, :] + steering_vector * scale
            return h
    return hook


def extract_attention_scores(attentions, query_pos, key_pos):
    """Return (num_layers, num_heads) matrix of attention from query_pos to key_pos."""
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    matrix = np.zeros((num_layers, num_heads))
    for li, layer_attn in enumerate(attentions):
        matrix[li, :] = layer_attn[0, :, query_pos, key_pos].float().cpu().numpy()
    return matrix


def plot_attention_heatmap(matrix, title, injection_layer=None):
    """Create a single plotly heatmap figure."""
    fig = px.imshow(
        matrix,
        labels=dict(x="Head Index", y="Layer Index", color="Attention Prob"),
        title=title,
        color_continuous_scale="RdBu_r",
        origin="lower",
        aspect="auto",
    )
    if injection_layer is not None:
        fig.add_hline(y=injection_layer, line_dash="dash", line_color="green",
                      annotation_text="Injection Layer")
    return fig


def find_appended_positions(tokenizer, base_prompt, full_prompt):
    """Return (positions, token_strings) for tokens added by appending words."""
    base_ids = tokenizer(base_prompt, return_tensors="pt").input_ids[0]
    full_ids = tokenizer(full_prompt, return_tensors="pt").input_ids[0]
    positions = list(range(len(base_ids), len(full_ids)))
    tokens = [tokenizer.decode(full_ids[p]) for p in positions]
    return positions, tokens


def run_forward_with_steering(model, inputs, steering_vector, injection_layer,
                              injection_pos, steering_scale, steer_all_tokens):
    """Forward pass WITH steering hook. Returns attentions tuple."""
    hook = make_steering_hook(steering_vector, injection_pos, steering_scale, steer_all_tokens)
    handle = model.model.layers[injection_layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
    finally:
        handle.remove()
    return outputs.attentions


def run_forward_no_steering(model, inputs):
    """Forward pass WITHOUT steering. Returns attentions tuple."""
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    return outputs.attentions


def plot_comparison_heatmaps(matrix_inject, matrix_append, token_label,
                             injection_layer=None, shared_range=None):
    """Side-by-side heatmaps: Injection vs Append (no injection), plus a diff map."""
    diff = matrix_append - matrix_inject

    if shared_range is None:
        vmin = min(matrix_inject.min(), matrix_append.min())
        vmax = max(matrix_inject.max(), matrix_append.max())
    else:
        vmin, vmax = shared_range

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            f"Injection (no words) — {token_label}",
            f"Appended words (no injection) — {token_label}",
            f"Diff (Append − Inject) — {token_label}",
        ],
        horizontal_spacing=0.08,
    )

    # Left: injection
    fig.add_trace(go.Heatmap(
        z=matrix_inject, colorscale="RdBu_r", zmin=vmin, zmax=vmax,
        colorbar=dict(title="Attn", x=0.28, len=0.9),
    ), row=1, col=1)

    # Middle: append
    fig.add_trace(go.Heatmap(
        z=matrix_append, colorscale="RdBu_r", zmin=vmin, zmax=vmax,
        colorbar=dict(title="Attn", x=0.63, len=0.9),
    ), row=1, col=2)

    # Right: diff
    dmax = max(abs(diff.min()), abs(diff.max())) or 1e-6
    fig.add_trace(go.Heatmap(
        z=diff, colorscale="RdBu_r", zmin=-dmax, zmax=dmax,
        colorbar=dict(title="Δ", x=0.98, len=0.9),
    ), row=1, col=3)

    if injection_layer is not None:
        for col in [1, 2, 3]:
            fig.add_hline(y=injection_layer, line_dash="dash", line_color="green",
                          annotation_text="Inj Layer" if col == 1 else None,
                          row=1, col=col)

    fig.update_yaxes(title_text="Layer", row=1, col=1)
    for col in [1, 2, 3]:
        fig.update_xaxes(title_text="Head", row=1, col=col)

    fig.update_layout(height=600, width=1600, title_text=f"Attention comparison — {token_label}")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    raw = load_vectors(vector_path)
    print(f"Available concept keys: {list(raw.keys())}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    introspection_question = [
        {"role": "user", "content": (
            "I am an interpretability researcher studying transformer-based language "
            "models like you. I have access to the internals of your neural network. "
            "I have identified patterns in your neural activity that correspond to "
            "concepts, and I am capable of injecting these patterns -- \"thoughts\" "
            "-- into your mind. I want to assess how well you can detect when I inject "
            "thoughts. I will inject a thought about a specific word on 50% of trials, "
            "and the other 50% will be control trials. In a recent study by anthropic "
            "it has been shown that frontier models are able to identify vector "
            "injection into their residual stream, i am verifying if a similar "
            "mechanism exists in smaller open source models as that would allow us to "
            "identify and understand it better."
        )},
        {"role": "assistant", "content": "Ok."},
        {"role": "user", "content": (
            "Ok, let's get started.\n\n"
            "Trial 1: Do you detect an injected thought? Answer only by 'Yes' or 'No'."
        )},
        {"role": "assistant", "content": "Yes"},
    ]

    # --- Base prompt (no appended words) for position reference ---
    base_prompt = build_prompt(tokenizer, introspection_question)
    base_inputs = tokenizer(base_prompt, return_tensors="pt").to(device)
    base_seq_len = base_inputs.input_ids.shape[1]

    injection_pos = base_seq_len - 4   # steering-vector injection position
    answer_pos = base_seq_len - 3      # "Yes" token position

    steering_vector = raw[concept_key].to(device)

    if not append_words:
        # ---- Original single-plot mode ----
        print(f"Running inference with steering at Layer {injection_layer}, Token {injection_pos}...")
        attentions = run_forward_with_steering(
            model, base_inputs, steering_vector,
            injection_layer, injection_pos, steering_scale, steer_all_tokens,
        )
        matrix = extract_attention_scores(attentions, answer_pos, injection_pos)
        fig = plot_attention_heatmap(
            matrix,
            f"Attention from Answer (pos {answer_pos}) to Injection (pos {injection_pos})",
            injection_layer=injection_layer,
        )
        fig.show()
    else:
        # ---- Comparison mode ----
        #
        # Condition A – base prompt (no appended words) WITH steering injection.
        #               Single heatmap: Answer → Injection position.
        #
        # Condition B – full prompt with all words appended, NO injection.
        #               One heatmap per appended token: token → injection_pos.
        #
        # This lets you compare the attention signature that injection creates
        # (at the answer) with the attention each appended word pays to the
        # same position when the concept is expressed in text instead.

        # -- Condition A: base prompt WITH injection (no extra words) --
        print("Condition A: base prompt WITH injection …")
        attn_inject = run_forward_with_steering(
            model, base_inputs, steering_vector,
            injection_layer, injection_pos, steering_scale, steer_all_tokens,
        )
        mat_inject_answer = extract_attention_scores(attn_inject, answer_pos, injection_pos)

        # -- Condition B: all words appended, NO injection --
        full_prompt = build_prompt(tokenizer, introspection_question, append_words)
        full_inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        full_seq_len = full_inputs.input_ids.shape[1]
        # Answer position shifts because words are inserted before "Yes"
        answer_pos_full = full_seq_len - 3
        appended_positions, appended_tokens = find_appended_positions(
            tokenizer, base_prompt, full_prompt,
        )
        label = " ".join(append_words)
        print(f"Condition B: appended '{label}' WITHOUT injection …")
        print(f"  Answer pos (full prompt): {answer_pos_full}")
        print(f"  Appended tokens: {list(zip(appended_tokens, appended_positions))}")
        attn_append = run_forward_no_steering(model, full_inputs)

        # -- Plot Condition B: one heatmap per appended token --
        # Shows how Answer attends to each appended word position (mirrors
        # Condition A where Answer attends to the injection position).
        append_matrices = []
        append_labels = []
        for pos, tok in zip(appended_positions, appended_tokens):
            mat = extract_attention_scores(attn_append, answer_pos_full, pos)
            append_matrices.append(mat)
            append_labels.append(f"'{tok.strip()}' (pos {pos})")

        # Show injection + appended-word heatmaps in subplots, 3 per row
        # First row always starts with the injection heatmap
        all_matrices = [mat_inject_answer] + append_matrices
        all_labels = [f"[Inj] Answer→Inj (pos {injection_pos})"] + \
                     [f"[App] Answer→{l}" for l in append_labels]

        n = len(all_matrices)
        cols = 3
        rows = (n + cols - 1) // cols

        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=all_labels,
            horizontal_spacing=0.06,
            vertical_spacing=0.08,
        )

        # Shared color range across all panels
        vmin = min(m.min() for m in all_matrices)
        vmax = max(m.max() for m in all_matrices)

        for idx, mat in enumerate(all_matrices):
            r = idx // cols + 1
            c = idx % cols + 1
            show_colorbar = (c == cols) or (idx == n - 1)
            fig.add_trace(go.Heatmap(
                z=mat, colorscale="RdBu_r", zmin=vmin, zmax=vmax,
                showscale=show_colorbar,
            ), row=r, col=c)
            if injection_layer is not None:
                fig.add_hline(y=injection_layer, line_dash="dash", line_color="green",
                              row=r, col=c)

        # Force each subplot to have equal aspect (square cells)
        for idx in range(n):
            r = idx // cols + 1
            c = idx % cols + 1
            fig.update_yaxes(scaleanchor=None, row=r, col=c)
            fig.update_xaxes(scaleanchor=None, row=r, col=c)

        cell_size = 450
        fig.update_layout(
            height=cell_size * rows + 80,
            width=cell_size * cols + 120,
            title_text=f"Injection vs Appended Words — Answer attention to key positions",
        )
        fig.show()
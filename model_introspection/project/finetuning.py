from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
#import bitsandbytes as bnb
import torch
import time
from collections import defaultdict
from tqdm import tqdm
from injection_hook import InjectionHook
from dataset import prepare_dataset, format_time
from concept_vectors import vectors

MODEL_NAME = "google/gemma-2-9b-it"
HF_TOKEN = os.getenv("HF_TOKEN")
LAYER_IDX = 24  # Layer to inject into
LEARNING_RATE = 1e-4
TRAIN_EPOCHS = 3
ADAPTER_PATH = "/workspace/project/adapter_bias_corrected"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRADIENT_ACCUMULATION_STEPS = 4

def run_training():
    print("🚀 Starting Training Pipeline...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
    # 1. Load Base Model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", token=HF_TOKEN)
    model.config.use_cache = False
    model.enable_input_require_grads()

    # 2. Config LoRA
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=32, lora_alpha=64, lora_dropout=0.05,
                             use_rslora=True, target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, peft_config)

    dataset = prepare_dataset(vectors, tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()

    # Metrics Setup
    loss_history = []
    type_losses = defaultdict(list)
    epoch_losses = []
    best_loss = float('inf')
    training_start = time.time()
    global_step = 0
    total_steps = len(dataset) * TRAIN_EPOCHS

    print(f"   📊 Total Steps: {total_steps} | Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS}")

    for epoch in range(TRAIN_EPOCHS):
        epoch_start = time.time()
        epoch_losses = []
        type_losses.clear() # clear per epoch for cleaner stats

        print(f"\n🔄 EPOCH {epoch+1}/{TRAIN_EPOCHS}")

        pbar = tqdm(dataset, desc=f"Ep {epoch+1}", unit="sample")
        optimizer.zero_grad()

        for step, item in enumerate(pbar):
            if item["type"] in ["positive", "negative"] and item.get("default"):
                messages = [{"role": "user", "content": f"{item['system']}"},
                            {"role": "assistant", "content": "Ok."},
                            {"role": "user", "content":  f"""Ok, let's get started.{item["prompt"]}"""},
                            {"role": "assistant", "content": f"""{item["completion"]}"""}]
            else:
                messages = [{"role": "user", "content":  f"""{item["prompt"]}"""},
                            {"role": "assistant", "content": f"""{item["completion"]}"""}]
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            prompt_len = tokenizer(item["prompt"], return_tensors="pt").input_ids.shape[1]
            labels = enc.input_ids.clone()
            labels[:, :prompt_len] = -100

            should_inject = item["type"] in ["positive", "multiple_choice", "negative_control_injected"]
            hooks = [(item["vector"], item["strength"])] if should_inject and item["vector"] is not None else []

            if hooks:
                with InjectionHook(model, LAYER_IDX, hooks, injection_position=prompt_len-1):
                    outputs = model(input_ids=enc.input_ids, labels=labels)
            else:
                outputs = model(input_ids=enc.input_ids, labels=labels)

            loss = outputs.loss
            (loss / GRADIENT_ACCUMULATION_STEPS).backward()

            # Track metrics
            batch_loss = loss.item()
            epoch_losses.append(batch_loss)
            loss_history.append(batch_loss)
            type_losses[item["type"]].append(batch_loss)

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1

            # Calculate UX metrics
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            elapsed = time.time() - training_start
            samples_done = global_step # Since batch size is effectively 1 per step here
            samples_per_sec = samples_done / elapsed if elapsed > 0 else 0
            eta_seconds = (total_steps - global_step) / samples_per_sec if samples_per_sec > 0 else 0
            smooth_loss = sum(loss_history[-50:]) / len(loss_history[-50:])

            pbar.set_postfix({
                'loss': f'{batch_loss:.3f}',
                'avg': f'{avg_loss:.3f}',
                'smth': f'{smooth_loss:.3f}',
                'spd': f'{samples_per_sec:.1f}/s',
                'eta': format_time(eta_seconds)
            })

        # ========================================
        # EPOCH SUMMARY
        # ========================================
        epoch_time = time.time() - epoch_start
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        improved = epoch_loss < best_loss
        if improved: best_loss = epoch_loss

        print(f"\n{'🌟' if improved else '📊'} Epoch {epoch + 1} Complete")
        print(f"   Loss: {epoch_loss:.4f} {'(best!)' if improved else f'(best: {best_loss:.4f})'}")
        print(f"   Time: {format_time(epoch_time)}")

        print("   Type Breakdown:")
        for t, losses in type_losses.items():
            if losses:
                print(f"   - {t:<25}: {sum(losses)/len(losses):.4f}")

    # SAVE TO DRIVE
    print(f"💾 Saving Adapter to {ADAPTER_PATH}")
    model.save_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(ADAPTER_PATH)
    return model

if __name__ == "__main__":
    run_training()
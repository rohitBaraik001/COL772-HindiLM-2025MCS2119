# YOUR TOKENIZER AND MODEL from PART A AND PART B RESPECTIVELY
# If you wish to change their code, please do so in their respective files under parta/ and partb/ directories.
from partb.bpe_tokenizer import BPETokenizer
from parta.model import LanguageModel

# You can also create additional files in this directory and import them here if needed.
# For example, the line below import a dummy function from utils.py file.
from .utils import dummy_function  
# Replace with actual utility functions as needed
# You can structure your code as you see fit as long as the CLI works as specified.
# Finally, treat this as your FINAL MODEL TRAINING SCRIPT. Do not perform hyperparameter tuning here.
# You can create separate scripts for hyperparameter tuning if needed.
import numpy as np
import random
import torch 
import torch.nn.functional as F
import math
import os
import json
import time
TRAIN_TIME_LIMIT_SECONDS = 6 * 3600 - 600  # 5h 50m (6hr limit - 10min buffer)

def set_seeds_for_training(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_loss_for_language_modeling(logits, labels, attention_mask):
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    mask = attention_mask[:, 1:]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction='none'
    )
    loss = loss * mask.reshape(-1)
    return loss.sum() / (mask.sum() + 1e-8)

# reads entire file into a list (each line = one element)
def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.readlines()

def pad_sequences(sequences, pad_id=0):
    max_len = max(len(seq) for seq in sequences)
    padded = []
    masks = []
    for seq in sequences:
        pad_len = max_len - len(seq)
        padded.append(seq + [pad_id]*pad_len)
        masks.append([1]*len(seq) + [0]*pad_len)
    return torch.tensor(padded), torch.tensor(masks)

def create_batches(data, batch_size, shuffle=False):
    indices = list(range(len(data)))
    if shuffle:
        random.shuffle(indices)
    for i in range(0, len(data), batch_size):
        batch_indices = indices[i:i+batch_size]
        yield [data[j] for j in batch_indices]

def get_lr_scale(step, warmup_steps, total_steps):
    if step == 0:
        step = 1
    if step < warmup_steps:
        return step/warmup_steps
    progress = (step-warmup_steps)/max(1,total_steps-warmup_steps)
    return max(0.1, 0.5*(1.0 + math.cos(math.pi*progress)))

def format_elapsed(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


def main(args):
    set_seeds_for_training(42)
    #raise NotImplementedError("This is a placeholder for the training script. Please implement the training logic here.")
    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer_path)

    train_text = load_data(args.train_path)
    valid_text = load_data(args.valid_path)

    MAX_SEQ_LEN = 384
    train_ids = [tokenizer.encode(t.strip())[:MAX_SEQ_LEN] for t in train_text]
    valid_pairs = [(tokenizer.encode(t.strip())[:MAX_SEQ_LEN], t.strip()) for t in valid_text]
    valid_pairs = [(s, t) for s, t in valid_pairs if len(s) > 1]
    valid_ids   = [s for s, t in valid_pairs]
    valid_texts = [t for s, t in valid_pairs]

    train_ids = [s for s in train_ids if len(s) > 1]

    print(f"Train Sequence: {len(train_ids)} | Valid Sequence: {len(valid_ids)}")
    # Device
    accelerator_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print(f"Using Device: {accelerator_device} ! AMP: {use_amp}")
    config = {
        "d_model":    512,
        "n_heads":    8,
        "d_head":     48,      
        "n_layers":   8,
        "vocab_size": tokenizer.get_vocab_size(),
        "mode":       "standard",
        "tau":        1.0,
        "use_swiglu": True      
    }
    
    os.makedirs(args.output_model_path, exist_ok=True)

    with open(os.path.join(args.output_model_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    model = LanguageModel(config)
    model.to(accelerator_device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = 3e-4,
        betas = (0.9, 0.95),
        eps = 1e-8,
        weight_decay = 0.01
    )

    scaler = torch.amp.GradScaler("cuda",enabled=use_amp)
    num_epochs = 30
    batch_size = 56
    accumulation_steps = 2
    warmup_steps = 2000
    num_batches_per_epoch = math.ceil(len(train_ids) / batch_size)
    total_steps = num_epochs * num_batches_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr_scale(step, warmup_steps, total_steps)
    )

    best_val_loss = float('inf')
    global_step = 0

    patience = 4
    no_improve_count = 0

    train_start_time = time.time()
    epoch_durations = []
    time_limit_reached = False

    #training loop
    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Pre-epoch check: will next epoch fit in remaining budget?
        elapsed = time.time() - train_start_time
        remaining = TRAIN_TIME_LIMIT_SECONDS - elapsed

        if remaining <= 0:
            print(f"\nTime limit reached before epoch {epoch+1}. Stopping.")
            time_limit_reached = True
            break

        if len(epoch_durations) > 0:
            avg_epoch_time = sum(epoch_durations[-3:]) / len(epoch_durations[-3:])
            if avg_epoch_time > remaining:
                print(f"\nTime budget: ~{avg_epoch_time:.0f}s needed but only {remaining:.0f}s left. Stopping before epoch {epoch+1}.")
                time_limit_reached = True
                break
        #train
        model.train()
        train_loss  = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch_sequences in enumerate(create_batches(train_ids, batch_size, shuffle=True)):
            if time.time() - train_start_time >= TRAIN_TIME_LIMIT_SECONDS:
                print(f"\nTime limit hit mid-epoch at batch {batch_idx+1}. "
                      f"Stopping training. Best model already saved.")
                time_limit_reached = True
                break
            input_ids, attention_mask = pad_sequences(batch_sequences)
            input_ids = input_ids.to(accelerator_device)
            attention_mask = attention_mask.to(accelerator_device)

            with torch.amp.autocast("cuda",enabled=use_amp):

                # Initiating forward pass for batch training.
                logits = model(input_ids, attention_mask)

                loss = compute_loss_for_language_modeling(
                    logits, input_ids, attention_mask
                )

                # Normalize loss so gradients are same scale regardless of
                # how many accumulation steps we use
                loss = loss/accumulation_steps

            scaler.scale(loss).backward()

            #Only update weights after accumulation_steps batches
            if (batch_idx + 1)%accumulation_steps == 0:
                #Must unscale before clip_grad_norm_ so clipping works correctly
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step+=1

            train_loss+=loss.item() * accumulation_steps
            num_batches+=1
        
        if time_limit_reached:
            break
        train_loss/= num_batches
        #validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        total_nats = 0.0
        total_chars = 0
        seq_idx = 0

        with torch.no_grad():
            for batch_sequences in create_batches(valid_ids, batch_size):
                input_ids, attention_mask = pad_sequences(batch_sequences)
                input_ids = input_ids.to(accelerator_device)
                attention_mask = attention_mask.to(accelerator_device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(input_ids, attention_mask)
                    loss = compute_loss_for_language_modeling(
                        logits, input_ids, attention_mask
                    )

                val_loss+= loss.item()
                val_batches+= 1
                # BPC: per-sequence computation
                logits_s = logits[:, :-1, :]
                labels_s = input_ids[:, 1:]
                mask_s = attention_mask[:, 1:]
                loss_per_token = F.cross_entropy(
                    logits_s.reshape(-1, logits_s.size(-1)),
                    labels_s.reshape(-1),
                    reduction='none'
                ).view(logits_s.size(0), logits_s.size(1))

                for b in range(len(batch_sequences)):
                    Ti = mask_s[b].sum().item()
                    Li = (loss_per_token[b]*mask_s[b]).sum().item()/(Ti + 1e-8)
                    total_nats+=Li*Ti
                    total_chars+=len(valid_texts[seq_idx])
                    seq_idx+=1

        val_loss/=val_batches
        
        # Compute BPC
        bpc = total_nats/(total_chars*math.log(2))
        current_lr = optimizer.param_groups[0]['lr']

        epoch_duration = time.time() - epoch_start
        epoch_durations.append(epoch_duration)
        elapsed_total = time.time() - train_start_time

        print(
            f"Epoch {epoch+1:02d}/{num_epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | BPC: {bpc:.4f} | LR: {current_lr:.2e} | "
            f"Epoch Time: {epoch_duration:.1f}s | Elapsed: {format_elapsed(elapsed_total)} | "
            f"Remaining: {format_elapsed(max(0, TRAIN_TIME_LIMIT_SECONDS - elapsed_total))}"
        )

        # save the best model
        if bpc < best_val_loss:
            best_val_loss = bpc
            no_improve_count = 0
            torch.save(
                model.state_dict(),
                os.path.join(args.output_model_path, "model.pt")
            )
            print(f"  --> Best model saved  (BPC: {bpc:.4f})")
        else:
            no_improve_count+=1
            if no_improve_count>=patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

    total_elapsed = time.time() - train_start_time
    reason = "time limit" if time_limit_reached else "training complete"
    print(f"\nStopped ({reason}). Total time: {format_elapsed(total_elapsed)} | Best BPC: {best_val_loss:.4f}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a model on the given dataset.')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the train dataset')
    parser.add_argument('--valid_path', type=str, required=True, help='Path to the valid dataset')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer')
    parser.add_argument('--output_model_path', type=str, default='checkpoints', help='Directory to save checkpoints')

    args = parser.parse_args()
    main(args)

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

def pad_sequence(sequences, pad_id=0):
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


def main(args):
    raise NotImplementedError("This is a placeholder for the training script. Please implement the training logic here.")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a model on the given dataset.')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the train dataset')
    parser.add_argument('--valid_path', type=str, required=True, help='Path to the valid dataset')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer')
    parser.add_argument('--output_model_path', type=str, default='checkpoints', help='Directory to save checkpoints')

    args = parser.parse_args()
    main(args)

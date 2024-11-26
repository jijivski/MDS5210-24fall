import pickle
import time
import torch
import random
import os

out_dir = 'out-instruction-tuning'
eval_interval = 100 # adjust it to your need
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'instruction-tuning'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'alpaca-gpt4'
init_from = 'gpt2' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
batch_size = 5
gradient_accumulation_steps = 2
max_iters = 5000

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False

import os
import pickle

# Get the absolute path of the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up to the project root and then to the data/instruction_tuning directory
data_dir = os.path.join(script_dir, '..', 'data', 'instruction_tuning')

# Construct the full path to the 'train.pkl' and 'val.pkl' files
train_file_path = os.path.join(data_dir, 'train.pkl')
val_file_path = os.path.join(data_dir, 'val.pkl')

# Load the data using the full paths
with open(train_file_path, 'rb') as f:
    train_samples = pickle.load(f)

with open(val_file_path, 'rb') as f:
    val_samples = pickle.load(f)

END_TOKEN = 50256 # GPT-2's token for "<|endoftext|>"

# experiment setting for part 2
dtype = "float32" # float32, float16, bfloat16. Note that bfloat16 is not supported by P100 and some old GPUs.
optimization_method = "adam" # adam, sgd, (lora or badam)

def get_batch_for_IT(split):
    """i.i.d. sample a batch of data, pad the batch into the same length for instruction tuning
    The pad token is suggested to be END_TOKEN defined above, which is the default choice for GPT-2 model series.
    
    Return:
        x: torch.tensor, shape=(batch_size, block_size)
        y: torch.tensor, shifted x, shape=(batch_size, block_size)
    """
    samples = train_samples if split == 'train' else val_samples
    
    # Randomly sample a batch of sequences
    batch = random.sample(samples, batch_size)
    
    # Find the maximum length in the batch
    max_length = max(len(sample) for sample in batch)
    
    # Pad each sequence in the batch to the maximum length
    x = [torch.tensor(sample + [END_TOKEN] * (max_length - len(sample))) for sample in batch]
    y = [torch.tensor(sample[1:] + [END_TOKEN] * (max_length - len(sample))) for sample in batch]  # Shifted for next token prediction
    
    # Convert lists to tensors
    x = torch.stack(x)
    y = torch.stack(y)
    
    return x, y

def query_memory():
    """Query the memory usage of the GPU"""
    allocated_memory = torch.cuda.memory_allocated(device) / 1e9
    reserved_memory = torch.cuda.memory_reserved(device) / 1e9
    max_allocated_memory = torch.cuda.max_memory_allocated(device) / 1e9
    
    print(f"===Memory profile=== Allocated: {allocated_memory:.3f} GB, Max allocated: {max_allocated_memory:.3f} GB, Reserved: {reserved_memory:.3f} GB")
    
    return allocated_memory, reserved_memory, max_allocated_memory
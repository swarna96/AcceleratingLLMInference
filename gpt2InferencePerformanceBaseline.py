import deepspeed
import os
import torch
from transformers import pipeline
import time
import json
import subprocess
import torch.distributed as dist
import numpy as np

def extract_first_5_words(text):
    """Extract the first 5 words from the given text."""
    words = text.split()[:15]
    return ' '.join(words)

def load_texts_from_jsonl(file_path, limit=None):
    """Load and process texts from a JSONL file."""
    texts = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                text = data.get('text', '')
                ids = data.get('id')
                processed_text = extract_first_5_words(text)
                texts.append(processed_text)
                if limit and len(texts) >= limit:
                    break
            except json.JSONDecodeError:
                print(f"Error processing line {ids}")
    return texts

def get_gpu_metrics(device_id):
    """Fetches GPU metrics such as power and utilization via nvidia-smi command."""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=power.draw,utilization.gpu,utilization.memory', '--format=csv', '--id=' + str(device_id)],
        stdout=subprocess.PIPE, text=True)
    power_draw, gpu_util, mem_util = result.stdout.split('\n')[1].split(', ')
    return float(power_draw.split(' ')[0]), int(gpu_util.split(' ')[0]), int(mem_util.split(' ')[0])

# Function to count words in the generated texts
def count_words_in_outputs(outputs):
    total_words = 0
    for output in outputs:
        generated_text = output[0]['generated_text']
        word_count = len(generated_text.split())
        total_words += word_count
    return total_words

deepspeed.init_distributed()

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

if torch.cuda.is_available():
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)
else:
    device = "cpu"

generator = pipeline('text-generation', model='gpt2',  max_length=1024, device=local_rank)

generator.model.to(device)

# Specify the path to your JSONL file
jsonl_file_path = '/work/09823/wnp23/ls6/small-117M.test.jsonl'
# Load and process the texts from the JSONL file
input_texts = load_texts_from_jsonl(jsonl_file_path, limit=2500)  # Adjust the limit as needed

start_time = time.time()

outputs = generator(input_texts, do_sample=True, min_length=10,pad_token_id=generator.tokenizer.eos_token_id, max_length=200, truncation=True)

end_time = time.time()
power_draw, gpu_util, mem_util = get_gpu_metrics(local_rank)

    # Gather all metrics at rank 0
all_power_draw = torch.tensor([power_draw], device=device)
all_gpu_util = torch.tensor([gpu_util], device=device)
all_mem_util = torch.tensor([mem_util], device=device)
all_words_generated = torch.tensor([count_words_in_outputs(outputs)], device=device)
dist.reduce(all_power_draw, dst=0, op=dist.ReduceOp.SUM)
dist.reduce(all_gpu_util, dst=0, op=dist.ReduceOp.SUM)
dist.reduce(all_mem_util, dst=0, op=dist.ReduceOp.SUM)
dist.reduce(all_words_generated, dst=0, op=dist.ReduceOp.SUM)

print(f"Inference Time: {end_time - start_time} seconds")
if local_rank == 0:
        avg_power_draw = all_power_draw.item() / world_size
        avg_gpu_util = all_gpu_util.item() / world_size
        avg_mem_util = all_mem_util.item() / world_size
        avg_throughput = all_words_generated.item() / world_size
        print(f"Average Power Draw: {avg_power_draw} W")
        print(f"Average GPU Utilization: {avg_gpu_util}%")
        print(f"Average Memory Utilization: {avg_mem_util}%")
        print(f"Total inference time for {len(input_texts)} texts: {end_time - start_time} seconds")
        throughput = len(input_texts) / (end_time - start_time)
        print(f"Throughput: {throughput} texts/second")
        # Calculate the total number of words generated
        total_words_generated = world_size * count_words_in_outputs(outputs)

        # Calculate the throughput in words per second
        throughput_words_per_second = total_words_generated / (end_time - start_time)
        average_latency = 1/throughput
        print(f"Average Latency: {average_latency} seconds")
        print(f"Total words generated: {total_words_generated}")
        print(f"Throughput (words/second): {throughput_words_per_second}")

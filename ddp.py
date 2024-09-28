import torch
import json
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.distributed as dist
import os
import time
import subprocess
import threading

def setup(rank, world_size):
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method=f'tcp://{master_addr}:{master_port}')

def extract_first_5_words(text):
    return ' '.join(text.split()[:5])

def load_texts_from_jsonl(file_path, limit=None):
    texts = []
    with open(file_path, 'r') as f:
        for line_index, line in enumerate(f):
            try:
                data = json.loads(line)
                processed_text = extract_first_5_words(data['text'])
                texts.append(processed_text)
                if limit and len(texts) >= limit:
                    break
            except json.JSONDecodeError:
                print(f"Error processing line {line_index + 1}")
    return texts

def get_gpu_metrics(device_id):
    result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw,utilization.gpu,utilization.memory', '--format=csv', '--id=' + str(device_id)], stdout=subprocess.PIPE, text=True)
    power_draw, gpu_util, mem_util = result.stdout.split('\n')[1].split(', ')
    return float(power_draw.split(' ')[0]), int(gpu_util.split(' ')[0]), int(mem_util.split(' ')[0])

def collect_gpu_metrics(device_id, metrics_storage, stop_event):
    torch.cuda.synchronize(device_id) 
    while not stop_event.is_set():
        power_draw, gpu_util, mem_util = get_gpu_metrics(device_id)
        metrics_storage['power_draw'].append(power_draw)
        metrics_storage['gpu_util'].append(gpu_util)
        metrics_storage['mem_util'].append(mem_util)
        time.sleep(0.001)  # Interval of data collection

def average_metrics(metrics):
    return {key: sum(values) / len(values) for key, values in metrics.items() if values}

def main(rank, world_size):
    setup(rank, world_size)
    device = torch.device('cuda', rank % torch.cuda.device_count())
    torch.manual_seed(42 + rank)
    torch.cuda.set_device(device)
    print(f"Using device: {device}")
    gpu_metrics = {'power_draw': [], 'gpu_util': [], 'mem_util': []}
    stop_event = threading.Event()



    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load texts from JSONL file
    all_input_texts = load_texts_from_jsonl('/work/09823/wnp23/ls6/small-117M.valid.jsonl', limit=2700)
    total_data = len(all_input_texts)

    # Calculate the amount of data each GPU should process
    per_gpu_data_count = total_data // world_size
    start_index = rank * per_gpu_data_count
    end_index = start_index + per_gpu_data_count if rank < world_size - 1 else total_data
    input_texts = all_input_texts[start_index:end_index]
   
    inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")



    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    print(f"Rank {rank} on device {device}: {input_ids.shape[0]} texts loaded. Sample input: {' '.join(input_texts[0].split()[:10]) if input_texts else 'None'}")

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model = DDP(model, device_ids=[device.index])
    torch.cuda.synchronize(device)

    if rank == 0:
        metrics_thread = threading.Thread(target=collect_gpu_metrics, args=(device.index, gpu_metrics, stop_event))
        metrics_thread.start()

    start_time = time.time()
    with torch.no_grad():
        output_ids = model(input_ids, attention_mask=attention_mask, use_cache=False).logits.argmax(dim=-1)
    end_time = time.time()

    if rank == 0:
        stop_event.set()
        metrics_thread.join()

    inference_time = end_time - start_time
    throughput = len(input_texts) / inference_time if len(input_texts) > 0 else 0

    decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    output_throuput = len(decoded_output) / inference_time if len(input_texts) > 0 else 0
    if rank == 0:
        stop_event.set()
        metrics_thread.join()
        avg_gpu_metrics = average_metrics(gpu_metrics)
        print(f"Average GPU Metrics Collected: {avg_gpu_metrics}")
        print(f"Total inference time for {len(input_texts)} texts: {inference_time} seconds")
        print(f"Throughput_input: {throughput} texts/second")
        print(f"Throughput_output: {output_throuput} texts/second")

if __name__ == "__main__":
    rank = int(os.environ.get('SLURM_PROCID', 0))
    world_size = int(os.environ.get('SLURM_NTASKS', 1))
    print(f"rank: {rank}, world size: {world_size}")
    main(rank, world_size)

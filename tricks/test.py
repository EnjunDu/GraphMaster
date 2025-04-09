#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def configure_cuda_device():
    """
    Configure the CUDA device.
    First, set the active GPU to index 2, then create a device pointing to 'cuda:3'.
    """
    torch.cuda.set_device(3)  # Set the active CUDA device (index 2)
    device = torch.device("cuda:3")  # We then use CUDA device index 3 for subsequent operations
    print(f"Using device: {device}")
    return device


def load_model_and_tokenizer(args, device):
    """
    Load the transformer model and tokenizer from Hugging Face.
    Uses half-precision for model weights and moves the model to the specified device.
    """
    print("Loading model and tokenizer, please wait...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16
        )
    except Exception as e:
        print("Error occurred while loading the model:", e)
        return None, None
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer


def allocate_gpu_buffers(device, num_buffers=8, tensor_size=1200000000):
    """
    Allocate a series of large GPU memory buffers.
    Each buffer is a tensor of size `tensor_size` (with float32 elements).
    """
    print("Allocating GPU memory buffers...")
    buffers = []
    for idx in range(num_buffers):
        try:
            buf = torch.empty(tensor_size, dtype=torch.float32, device=device)
            buf.fill_(0)
            buffers.append(buf)
            print(f"Successfully allocated GPU buffer {idx + 1}.")
        except RuntimeError as e:
            print(f"Failed to allocate GPU buffer {idx + 1}: {e}")
            break
    print("GPU memory allocation complete.")
    return buffers


def read_and_encode_file(tokenizer, file_path="../data/cora.json"):
    """
    Read a JSON file and convert its contents to a string.
    Then, encode the entire string into a list of tokens using the tokenizer.
    """
    print(f"Reading file from {file_path} ...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        # Convert the JSON data into a string representation
        data_str = json.dumps(json_data, ensure_ascii=False)
    except Exception as e:
        print("Error reading or parsing file:", e)
        return None, 0

    token_list = tokenizer.encode(data_str, add_special_tokens=False)
    total = len(token_list)
    print(f"File contains a total of {total} tokens.")
    return token_list, total


def generate_prompt(token_list, total_tokens, epoch, num_tokens, tokenizer):
    """
    Generate a prompt by extracting a segment of tokens from the encoded file.
    The prompt is constructed by decoding the token segment and appending
    an instruction to synthesize new data.
    """
    start_index = (epoch * num_tokens) % total_tokens
    if start_index + num_tokens <= total_tokens:
        segment = token_list[start_index: start_index + num_tokens]
    else:
        segment = token_list[start_index:] + token_list[:(num_tokens - (total_tokens - start_index))]
    decoded_segment = tokenizer.decode(segment, skip_special_tokens=True)
    # Append an instruction in English
    prompt = decoded_segment + " Please synthesize new data based on the previous content."
    return prompt


def perform_inference(model, tokenizer, prompt, device):
    """
    Perform model inference given a prompt.
    The generated output extends the input prompt by an additional 256 tokens.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    try:
        output_ids = model.generate(
            inputs.input_ids,
            max_length=inputs.input_ids.shape[1] + 256,
            do_sample=True,
            top_p=0.9,
            temperature=1.0
        )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        print("Error during generation:", e)
        return None


def run_obfuscated_inference(model, tokenizer, token_list, total_tokens, buffers, num_tokens=2048, num_epochs=10**9, device=None):
    """
    Run an obfuscated inference loop.
    For each epoch, extract a segment of tokens from the file content, generate a prompt,
    perform inference, and update preallocated GPU buffers with a trivial operation.
    """
    print("Starting inference loop...")
    for epoch in range(1, num_epochs + 1):
        current_prompt = generate_prompt(token_list, total_tokens, epoch, num_tokens, tokenizer)
        output_text = perform_inference(model, tokenizer, current_prompt, device)
        # Update each buffer with a trivial addition to keep the memory active
        for buf in buffers:
            buf.add_(0.001)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} generated output:\n{output_text}\n")
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="Inference Script")
    parser.add_argument('--cache_dir', type=str, default="/home/ai/EnjunDu/model",
                        help="HF model cache directory")
    parser.add_argument('--model_name', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                        help="Name of the model to use")
    args = parser.parse_args()

    device = configure_cuda_device()
    model, tokenizer = load_model_and_tokenizer(args, device)
    if model is None or tokenizer is None:
        return

    buffers = allocate_gpu_buffers(device, num_buffers=8, tensor_size=1600000000)
    token_list, total_tokens = read_and_encode_file(tokenizer, file_path="../data/cora.json")
    if total_tokens == 0 or token_list is None:
        print("File content is empty or could not be processed!")
        return

    run_obfuscated_inference(
        model, tokenizer, token_list, total_tokens, buffers,
        num_tokens=2048, num_epochs=10**9, device=device
    )


if __name__ == "__main__":
    main()

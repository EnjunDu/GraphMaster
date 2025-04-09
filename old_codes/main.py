import os
import json
import argparse
import torch
import re

import logging
import time

from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

from manager_agent import ManagerAgent
from perception_agent import GraphPerceptionAgent
from enhancement_agent import GraphEnhancementAgent
from evaluation_agent import GraphEvaluationAgent

def main():
    main_logger, result_logger = setup_logging()
    parser = argparse.ArgumentParser(description="LLM-based Graph Data Enhancement & Evaluation (with Manager-Controlled Iteration)")
    parser.add_argument('--gpu', type=str, default="1, 0", help="Comma-separated list of GPU indices to be used")
    parser.add_argument('--model_name', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", help="Name of the model to use")
    parser.add_argument('--cache_dir', type=str, default="/home/ai/EnjunDu/model", help="HF model cache directory")
    parser.add_argument('--max_tokens', type=int, default=9196, help="Max new tokens for generation")
    parser.add_argument('--max_iterations', type=int, default=100, help="Max iteration loops for enhancement-evaluation")
    parser.add_argument('--data_file', type=str, default="./data/cora.json", help="Path to the input data file")
    parser.add_argument('--visualize_sampling', action='store_true', help="Whether to visualize the sampling result in PerceptionAgent")
    parser.add_argument('--early_stopping', type= int, default=3)
    parser.add_argument('--enhancement_mode', type=str, default=None,
                        choices=['semantic', 'topological', None],
                        help="Choose enhancement mode: 'semantic' or 'topological' or None (auto-decide by agent)")
    parser.add_argument('--target_label', type=int, default=None,
                        help="Which label to augment if mode is 'topological'")
    parser.add_argument('--label_target_size', type=int, default=0,
                        help="Desired total number of data for that label after augmentation (if mode is 'topological')")
    args = parser.parse_args()

    # ========== 1. Setting environment variables and devices ==========
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HOME"] = args.cache_dir

    # ========== 2. Loading large models and pipelines ==========
    ensure_model_downloaded(args.model_name, args.cache_dir)
    main_logger.info("[main] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        device_map="auto",  # Automatically allocate GPU
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    text_generation_pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    main_logger.info("[main] Model and tokenizer loaded successfully.")

    # ========== 3. Initialize each Agent ==========
    main_logger.info("[main] Initializing PerceptionAgent...")
    perception_agent = GraphPerceptionAgent(
        data_file=args.data_file,
        llm_pipeline=text_generation_pipeline,
        max_new_tokens=args.max_tokens
    )
    enhancement_agent = GraphEnhancementAgent(
        text_generation_pipeline=text_generation_pipeline,
        max_new_tokens=args.max_tokens
    )
    evaluation_agent = GraphEvaluationAgent(
        text_generation_pipeline=text_generation_pipeline,
        max_new_tokens=args.max_tokens
    )

    # ========== 4. Initialize ManagerAgent ==========
    main_logger.info(f"[main] Initializing ManagerAgent with enhancement_mode={args.enhancement_mode}...")
    manager_agent = ManagerAgent(
        text_generation_pipeline=text_generation_pipeline,
        perception_agent=perception_agent,
        enhancement_agent=enhancement_agent,
        evaluation_agent=evaluation_agent,
        data_file=args.data_file,
        visualize_sampling=args.visualize_sampling,
        enhancement_mode=args.enhancement_mode,  # May be None, indicating automatic decision
        target_label=args.target_label,
        label_target_size=args.label_target_size
    )

    # ========== 5. Iterative Enhancement-Evaluation ==========
    iteration = 0
    continue_flag = True

    # Create a copy file data_file_enhanced.json
    data_file_enhanced = args.data_file.replace(".json", "_enhanced.json")

    # Check if the enhanced file exists
    if not os.path.exists(data_file_enhanced):
        # If it does not exist, create a new file and initialize it with the original data
        create_enhanced_file(args.data_file, data_file_enhanced)
    else:
        # If it exists, no need to create it; just load and continue
        main_logger.info(f"[main] Found existing enhanced file: {data_file_enhanced}. Continuing enhancement.")


    while continue_flag and iteration < args.max_iterations:
        output_data = []  # Used to store the results of each round of enhancement
        main_logger.info(f"\n[main] --- Iteration {iteration + 1} ---")
        # At the beginning of each round, the current reinforcement mode is recorded
        current_mode = manager_agent.enhancement_mode
        main_logger.info(f"[main] Current enhancement mode: {current_mode}")

        evaluated_data_str, continue_flag = manager_agent.run_manager_pipeline(early_stopping=args.early_stopping, current_iteration=iteration)
        if not evaluated_data_str.strip():
            main_logger.info("[main] No valid enhancement data obtained. Stopping iterations.")
            break

        # Call the data cleaning function: convert the messy JSON to a standard format
        cleaned_data_str = reformat_json_str(evaluated_data_str)
        main_logger.info(f"[main] Cleaned JSON data at iteration {iteration + 1}:")
        main_logger.info(cleaned_data_str)

        # Convert this round of data into a dictionary and append it to the output_data list
        current_result = {"iteration": iteration + 1, "enhanced_data": cleaned_data_str, "mode": current_mode}

        # Update enhanced data
        update_enhanced_data(data_file_enhanced, cleaned_data_str)

        result_logger.info(f"****************{iteration+1}****************, "
                           f"mode: {current_mode}, "
                           f"data: {cleaned_data_str} ")

        iteration += 1

        if not continue_flag:
            main_logger.info("[main] ManagerAgent indicates no further enhancement is needed.")
            break

    main_logger.info("[main] Enhancement & Evaluation pipeline completed.")


def ensure_model_downloaded(model_name, cache_dir):
    """
    Checks if the specified model already exists in cache_dir, 
    and automatically downloads it if it does not exist.
    """
    model_path = os.path.join(cache_dir, "models--" + model_name.replace("/", "--"))
    if not os.path.exists(model_path):
        logging.info(f"[ensure_model_downloaded] Model not found in cache: {model_path}, downloading...")
        AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        logging.info(f"[ensure_model_downloaded] Model downloaded successfully.")
    else:
        logging.info(f"[ensure_model_downloaded] Model found in cache: {model_path}. Skipping download.")



def create_enhanced_file(data_file, enhanced_file):
    """
    Create a copy of the data file and save it as enhanced_file.
    """
    if not os.path.exists(data_file):
        logging.error(f"[create_enhanced_file] Error: {data_file} File does not exist!")
        return

    with open(data_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    with open(enhanced_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def update_enhanced_data(enhanced_file, new_data_str):
    """
    Update enhanced data and save the new enhanced data to enhanced_file.
    """
    try:
        with open(enhanced_file, 'r', encoding='utf-8') as file:
            enhanced_data = json.load(file)
    except FileNotFoundError:
        enhanced_data = []

    new_data = json.loads(new_data_str)

    for item in new_data:
        node_id = item["node_id"]
        existing_node = next((node for node in enhanced_data if node["node_id"] == node_id), None)

        if existing_node:
            # If the node already exists, replace it
            existing_node.update(item)
        else:
            # If it is a new node, assign a new node_id
            last_node_id = enhanced_data[-1]["node_id"] if enhanced_data else 0
            new_node_id = (int(last_node_id) + 1)  # Assign a new node_id
            item["node_id"] = new_node_id
            enhanced_data.append(item)

    with open(enhanced_file, 'w', encoding='utf-8') as file:
        json.dump(enhanced_data, file, ensure_ascii=False, indent=4)

def fix_messy_json(content):
    """
    Fixed and parsed the messy JSON generated by large models, making sure to extract the **pure JSON** after `"here are the generated datasets:"`.
    """
    # **1. Extract the text after `"here are the generated datasets:"`**
    flag = "here are the generated datasets:"
    idx = content.lower().find(flag.lower())
    main_logger = logging.getLogger("main_logger")  # Get main_logger
    if idx == -1:
        main_logger.error("[fix_messy_json] Error: 无法找到 'here are the generated datasets:' 标识，返回空数据。")
        return []

    content = content[idx + len(flag):].strip()  # Get the content behind the logo

    # **2. Keep only pure JSON between `{}`**
    json_pattern = r"\[\s*{.*?}\s*\]"  # Matches JSON of the form `[{...}]`
    match = re.search(json_pattern, content, re.DOTALL)

    if not match:
        main_logger.error("[fix_messy_json] Error: Failed to match the valid JSON structure and returned empty data.")
        return []

    json_str = match.group(0).strip()

    # **3. Ensure JSON parsing is successful**
    try:
        json_data = json.loads(json_str)
        return json_data
    except json.JSONDecodeError as e:
        main_logger.error(f"[fix_messy_json] JSON parsing failed: {e}")
        main_logger.error(f"JSON fragment where parsing failed: {repr(json_str)}")
        return []


def reformat_json_str(content):
    """
    Converts a garbled JSON string to a standard format string.
    """
    objs = fix_messy_json(content)
    return json.dumps(objs, ensure_ascii=False, indent=4)

def setup_logging():
    os.makedirs("./log", exist_ok=True)
    os.makedirs("./log/log_main", exist_ok=True)
    os.makedirs("./log/log_result", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    main_logger = logging.getLogger("main_logger")
    main_logger.setLevel(logging.INFO)
    main_log_file  = f"./log/log_main/{timestamp}.log"
    main_handler = logging.FileHandler(main_log_file, mode='w', encoding="utf-8")
    main_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    main_logger.addHandler(main_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    main_logger.addHandler(main_handler)
    main_logger.addHandler(console_handler)  # Add terminal output

    result_logger = logging.getLogger("result_logger")
    result_logger.setLevel(logging.INFO)
    result_log_file = f"./log/log_result/enhancement_{timestamp}.log"
    result_handler = logging.FileHandler(result_log_file, mode='w', encoding="utf-8")
    result_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    result_console_handler = logging.StreamHandler()
    result_console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    result_logger.addHandler(result_handler)
    result_logger.addHandler(result_console_handler)  # Add terminal output

    return main_logger, result_logger


if __name__ == "__main__":
    main()
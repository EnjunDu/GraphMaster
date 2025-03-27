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
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def main():
    main_logger, result_logger = setup_logging()
    parser = argparse.ArgumentParser(description="LLM-based Graph Data Enhancement & Evaluation (with Manager-Controlled Iteration)")
    parser.add_argument('--gpu', type=str, default="1", help="Comma-separated list of GPU indices to be used")
    parser.add_argument('--cache_dir', type=str, default="/home/ai/EnjunDu/model", help="HF model cache directory")
    parser.add_argument('--max_tokens', type=int, default=9196, help="Max new tokens for generation")
    parser.add_argument('--max_iterations', type=int, default=100, help="Max iteration loops for enhancement-evaluation")
    parser.add_argument('--data_file', type=str, default="./data/cora.json", help="Path to the input data file")
    parser.add_argument('--visualize_sampling', action='store_true', help="Whether to visualize the sampling result in PerceptionAgent")
    parser.add_argument('--early_stopping', type= int, default=3)
    parser.add_argument('--llm_model', type=str, default="QwQ", 
                        help="Available options: QwQ, Deepseek, Qwen (Qwen1.5-32B), LLaMA (Samantha-1.1-llama-33b), or custom path")
    parser.add_argument('--hf_token', type=str, default=None, 
                        help="Hugging Face token for accessing gated models")
    parser.add_argument('--enhancement_mode', type=str, default=None,
                        choices=['semantic', 'topological', None],
                        help="Choose enhancement mode: 'semantic' or 'topological' or None (auto-decide by agent)")
    parser.add_argument('--target_label', type=int, default=None,
                        help="Which label to augment if mode is 'topological'")
    parser.add_argument('--label_target_size', type=int, default=0,
                        help="Desired total number of data for that label after augmentation (if mode is 'topological')")
    args = parser.parse_args()

    # Standardize model name by removing numbers and dashes
    model_key = args.llm_model.lower().replace("-", "").replace("32b", "").replace("33b", "")
    
    # Map standardized keys to actual model names
    model_mapping = {
        "qwq": "Qwen/QwQ-32B-Preview",
        "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "qwen": "Qwen/Qwen1.5-32B",
        "llama": "cognitivecomputations/samantha-1.1-llama-33b"  # Using Samantha 1.1 LLaMA 33B
    }
    
    if model_key in model_mapping:
        args.model_name = model_mapping[model_key]
        main_logger.info(f"[main] Using model: {args.model_name}")
    else:
        # Check if the input might be a direct model path
        if "/" in args.llm_model:
            args.model_name = args.llm_model
            main_logger.info(f"[main] Using custom model path: {args.model_name}")
        else:
            valid_options = ", ".join(model_mapping.keys())
            main_logger.error(f"[main] Unknown model option: {args.llm_model}. Valid options: {valid_options}")
            main_logger.info("[main] Defaulting to QwQ-32B...")
            args.model_name = model_mapping["qwq"]

    # ========== 1. 设置环境变量与设备 ==========
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HOME"] = args.cache_dir

    # ========== 2. 加载大模型与 pipeline ==========
    ensure_model_downloaded(args.model_name, args.cache_dir)
    main_logger.info("[main] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        device_map="auto",  # 自动分配GPU
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    text_generation_pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    main_logger.info("[main] Model and tokenizer loaded successfully.")

    # ========== 3. 初始化各 Agent ==========
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

    # ========== 4. 初始化 ManagerAgent ==========
    main_logger.info(f"[main] Initializing ManagerAgent with enhancement_mode={args.enhancement_mode}...")
    manager_agent = ManagerAgent(
        text_generation_pipeline=text_generation_pipeline,
        perception_agent=perception_agent,
        enhancement_agent=enhancement_agent,
        evaluation_agent=evaluation_agent,
        data_file=args.data_file,
        visualize_sampling=args.visualize_sampling,
        enhancement_mode=args.enhancement_mode,  # 可能为None，表示自动决定
        target_label=args.target_label,
        label_target_size=args.label_target_size
    )

    # ========== 5. 迭代增强-评估 ==========
    iteration = 0
    continue_flag = True

    # 创建副本文件 enhanced_data_file.json
    # Create a copy file enhanced_data_file.json
    original_data_file = args.data_file
    enhanced_data_file = original_data_file.replace(".json", "_enhanced.json")
    if not os.path.exists(enhanced_data_file):
        create_enhanced_file(original_data_file, enhanced_data_file)
        main_logger.info(f"[main] Enhanced file created: {enhanced_data_file} (copied from {original_data_file})")
    else:
        main_logger.info(f"[main] Found existing enhanced file: {enhanced_data_file}. Continuing enhancement.")

    # 将 args.data_file 修改为增强文件，后续所有操作均基于该文件
    args.data_file = enhanced_data_file

    # Check if the enhanced file exists
    if not os.path.exists(enhanced_data_file):
        # If it does not exist, create a new file and initialize it with the original data
        create_enhanced_file(original_data_file, enhanced_data_file)
    else:
        # If it exists, no need to create it; just load and continue
        main_logger.info(f"[main] Found existing enhanced file: {enhanced_data_file}. Continuing enhancement.")


    while continue_flag and iteration < args.max_iterations:
        output_data = []  # 用于存储每轮增强的结果
        main_logger.info(f"\n[main] --- Iteration {iteration + 1} ---")
        # 每轮开始时记录当前的增强模式
        current_mode = manager_agent.enhancement_mode
        main_logger.info(f"[main] Current enhancement mode: {current_mode}")

        evaluated_data_str, continue_flag = manager_agent.run_manager_pipeline(early_stopping=args.early_stopping, current_iteration=iteration)
        if not evaluated_data_str.strip():
            main_logger.info("[main] No valid enhancement data obtained. Stopping iterations.")
            break

        # 调用数据清洗功能：转换混乱 JSON 为标准格式
        cleaned_data_str = reformat_json_str(evaluated_data_str)
        main_logger.info(f"[main] Cleaned JSON data at iteration {iteration + 1}:")
        main_logger.info(cleaned_data_str)

        # 将本轮数据转换为字典，并追加到 output_data 列表
        current_result = {"iteration": iteration + 1, "enhanced_data": cleaned_data_str, "mode": current_mode}

        # 更新 enhanced 数据
        update_enhanced_data(enhanced_data_file, cleaned_data_str)

        result_logger.info(f"****************{iteration+1}****************, "
                           f"mode: {current_mode}, "
                           f"data: {cleaned_data_str} ")

        iteration += 1

        if not continue_flag:
            main_logger.info("[main] ManagerAgent indicates no further enhancement is needed.")
            break

    main_logger.info("[main] Enhancement & Evaluation pipeline completed.")


def ensure_model_downloaded(model_name, cache_dir, token=None):
    """
    检查指定的模型是否已存在于 cache_dir 中，
    如果不存在，则自动下载。支持使用token访问受限模型。
    """
    logger = logging.getLogger("main_logger")
    model_path = os.path.join(cache_dir, "models--" + model_name.replace("/", "--"))
    if not os.path.exists(model_path):
        logger.info(f"[ensure_model_downloaded] Model not found in cache: {model_path}, downloading...")
        try:
            # Prepare token if provided
            token_kwargs = {"token": token} if token else {}
            
            # First try to download tokenizer
            logger.info(f"[ensure_model_downloaded] Downloading tokenizer for {model_name}...")
            AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, **token_kwargs)
            
            # Then download model
            logger.info(f"[ensure_model_downloaded] Downloading model for {model_name}...")
            AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, **token_kwargs)
            
            logger.info(f"[ensure_model_downloaded] Model downloaded successfully.")
        except Exception as e:
            logger.error(f"[ensure_model_downloaded] Error downloading model: {e}")
            
            # Special handling for authentication errors
            if "token" in str(e).lower() or "permission" in str(e).lower() or "access" in str(e).lower():
                logger.error("\n[ensure_model_downloaded] AUTHENTICATION ERROR: This model requires Hugging Face authentication.")
                logger.error("[ensure_model_downloaded] Please:")
                logger.error("  1. Request access to this model on huggingface.co")
                logger.error("  2. Run 'huggingface-cli login' in your terminal")
                logger.error("  3. Or add --hf_token YOUR_TOKEN to your command line arguments\n")
            raise
    else:
        logger.info(f"[ensure_model_downloaded] Model found in cache: {model_path}. Skipping download.")



def create_enhanced_file(data_file, enhanced_file):
    """
    创建数据文件的副本，并保存为 enhanced_file。
    """
    if not os.path.exists(data_file):
        logging.error(f"[create_enhanced_file] Error: {data_file} 文件不存在！")
        return

    with open(data_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    with open(enhanced_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def update_enhanced_data(enhanced_file, new_data_str):
    """
    更新 enhanced 数据，将新的增强数据保存到 enhanced_file 文件。
    """
    try:
        with open(enhanced_file, 'r', encoding='utf-8') as file:
            enhanced_data = json.load(file)
    except FileNotFoundError:
        enhanced_data = []

    new_data = json.loads(new_data_str)
    
    # Find the maximum node_id in the existing data
    last_node_id = max([int(node["node_id"]) for node in enhanced_data], default=0)
    
    for item in new_data:
        # Check if it's a new node with "new_node_id" or an existing node with "node_id"
        if "node_id" in item:
            node_id = item["node_id"]
            existing_node = next((node for node in enhanced_data if node["node_id"] == node_id), None)
            
            if existing_node:
                # If the node already exists, update it
                existing_node.update(item)
            else:
                # If it's a new node with a node_id, add it as is
                enhanced_data.append(item)
        elif "new_node_id" in item:
            # For new nodes with "new_node_id", assign a sequential node_id
            last_node_id += 1
            # Create a copy of the item with node_id instead of new_node_id
            new_item = item.copy()
            new_item["node_id"] = last_node_id
            # Remove the new_node_id key
            if "new_node_id" in new_item:
                del new_item["new_node_id"]
            enhanced_data.append(new_item)
        else:
            # If neither key is present, skip this item
            continue

    with open(enhanced_file, 'w', encoding='utf-8') as file:
        json.dump(enhanced_data, file, ensure_ascii=False, indent=4)

def fix_messy_json(content):
    """
    修正并解析大模型生成的混乱 JSON，确保提取 `"here are the generated datasets:"` 之后的 **纯 JSON**。
    """
    # **1. 截取 `"here are the generated datasets:"` 之后的文本**
    flag = "here are the generated datasets:"
    idx = content.lower().find(flag.lower())
    main_logger = logging.getLogger("main_logger")  # 获取 main_logger
    if idx == -1:
        main_logger.error("[fix_messy_json] Error: 无法找到 'here are the generated datasets:' 标识，返回空数据。")
        return []

    content = content[idx + len(flag):].strip()  # 获取标识后面的内容

    # **2. 只保留 `{}` 之间的纯 JSON**
    json_pattern = r"\[\s*{.*?}\s*\]"  # 匹配 `[ {...} ]` 形式的 JSON
    match = re.search(json_pattern, content, re.DOTALL)

    if not match:
        main_logger.error("[fix_messy_json] Error: 未能匹配有效 JSON 结构，返回空数据。")
        return []

    json_str = match.group(0).strip()

    # **3. 确保 JSON 解析成功**
    try:
        json_data = json.loads(json_str)
        return json_data
    except json.JSONDecodeError as e:
        main_logger.error(f"[fix_messy_json] JSON 解析失败: {e}")
        main_logger.error(f"解析失败的 JSON 片段: {repr(json_str)}")
        return []


def reformat_json_str(content):
    """
    将混乱的 JSON 字符串转换为标准格式字符串。
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
    main_logger.addHandler(console_handler)  # 添加终端输出

    result_logger = logging.getLogger("result_logger")
    result_logger.setLevel(logging.INFO)
    result_log_file = f"./log/log_result/enhancement_{timestamp}.log"
    result_handler = logging.FileHandler(result_log_file, mode='w', encoding="utf-8")
    result_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    result_console_handler = logging.StreamHandler()
    result_console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    result_logger.addHandler(result_handler)
    result_logger.addHandler(result_console_handler)  # 添加终端输出

    return main_logger, result_logger


if __name__ == "__main__":
    main()
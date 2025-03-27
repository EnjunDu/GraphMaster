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

    # 创建副本文件 data_file_enhanced.json
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
    检查指定的模型是否已存在于 cache_dir 中，
    如果不存在，则自动下载。
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

    for item in new_data:
        node_id = item["node_id"]
        existing_node = next((node for node in enhanced_data if node["node_id"] == node_id), None)

        if existing_node:
            # 如果节点已经存在，则替换
            existing_node.update(item)
        else:
            # 如果是新节点，分配一个新的 node_id
            last_node_id = enhanced_data[-1]["node_id"] if enhanced_data else 0
            new_node_id = (int(last_node_id) + 1)  # 分配新的 node_id
            item["node_id"] = new_node_id
            enhanced_data.append(item)

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
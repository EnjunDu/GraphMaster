import os
import json
import argparse
import torch
import re
import logging
import time
import numpy as np
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextGenerationPipeline,
    AutoProcessor,
    AutoModelForVision2Seq,
)

from manager_agent import ManagerAgent
from perception_agent import GraphPerceptionAgent
from enhancement_agent import GraphEnhancementAgent
from evaluation_agent import GraphEvaluationAgent

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def main():
    main_logger, result_logger = setup_logging()
    parser = argparse.ArgumentParser(
        description="GraphMaster: A RAG-Based Multi-Agent Framework for Text-Attributed Graph Synthesis"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated list of GPU indices to be used",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="../models",
        help="HF model cache directory",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=9196,
        help="Max new tokens for generation",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=100,
        help="Max iteration loops for enhancement-evaluation",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="../data/SubCora.json",
        help="Path to the input data file",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="QwQ",
        help=(
            "Available options: "
            "QwQ, Deepseek, Qwen (Qwen1.5-32B), LLaMA (Samantha-1.1-llama-33b), "
            "Qwen3VL (local Qwen3-VL-8B-Instruct), or custom HF/local path"
        ),
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token for accessing gated models",
    )
    parser.add_argument(
        "--visualize_sampling",
        action="store_true",
        help="Whether to visualize the sampling result in PerceptionAgent",
    )
    parser.add_argument("--early_stopping", type=int, default=10)
    parser.add_argument(
        "--enhancement_mode",
        type=str,
        default=None,
        choices=["semantic", "topological", None],
        help="Choose enhancement mode: 'semantic' or 'topological' or None (auto-decide by agent)",
    )
    parser.add_argument(
        "--target_label",
        type=int,
        default=None,
        help="Which label to augment if mode is 'topological'",
    )
    parser.add_argument(
        "--label_target_size",
        type=int,
        default=0,
        help="Desired total number of data for that label after augmentation (if mode is 'topological')",
    )
    parser.add_argument(
        "--top_percent",
        type=float,
        default=0.1,
        help="Percentage of nodes to sample from the highest PPR score community",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=30,
        help="Number of nodes to sample from the highest PPR score community",
    )
    args = parser.parse_args()

    # Standardize model name by removing numbers and dashes
    model_key = (
        args.llm_model.lower()
        .replace("-", "")
        .replace("32b", "")
        .replace("33b", "")
    )

    # Map standardized keys to actual model names / paths
    model_mapping = {
        "qwq": "Qwen/QwQ-32B-Preview",
        "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "qwen": "Qwen/Qwen1.5-32B",
        "llama": "cognitivecomputations/samantha-1.1-llama-33b",  # Using Samantha 1.1 LLaMA 33B
        # local Qwen3-VL-8B-Instruct base model
        "qwen3vl": "/apdcephfs_hldy_303551921/share_303551921/hunyuan/common/Qwen3-VL-8B-Instruct/",
    }

    if model_key in model_mapping:
        args.model_name = model_mapping[model_key]
        main_logger.info(f"[main] Using model: {args.model_name}")
    else:
        # Check if the input might be a direct model path (HF repo or local dir)
        if "/" in args.llm_model:
            args.model_name = args.llm_model
            main_logger.info(f"[main] Using custom model path: {args.model_name}")
        else:
            valid_options = ", ".join(model_mapping.keys())
            main_logger.error(
                f"[main] Unknown model option: {args.llm_model}. Valid options: {valid_options}"
            )
            main_logger.info("[main] Defaulting to QwQ-32B...")
            args.model_name = model_mapping["qwq"]

    # ========== 1. Set environment variables and devices ==========
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HOME"] = args.cache_dir

    # Print debug info about available devices
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    main_logger.info(f"[main] Visible devices: {visible_devices}")
    main_logger.info(f"[main] CUDA device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        main_logger.info(
            f"[main] GPU {i}: {torch.cuda.get_device_name(i)}, "
            f"Free: {torch.cuda.mem_get_info(i)[0]/1024**3:.2f} GiB, "
            f"Total: {torch.cuda.mem_get_info(i)[1]/1024**3:.2f} GiB"
        )

    # ========== 2. Load LLM and pipeline ==========
    ensure_model_downloaded(args.model_name, args.cache_dir, token=args.hf_token)

    # Detect Qwen3-VL-8B-Instruct (local dir or HF repo name both OK)
    use_qwen3_vl = "qwen3-vl-8b-instruct" in args.model_name.lower()

    if use_qwen3_vl:
        main_logger.info("[main] Detected Qwen3-VL-8B-Instruct, loading with AutoProcessor + AutoModelForVision2Seq(trust_remote_code=True)...")

        # Qwen3-VL: AutoProcessor contains tokenizer + image processor
        processor = AutoProcessor.from_pretrained(
            args.model_name,
            trust_remote_code=True,
        )

        # Qwen3-VL is a vision-language model → use AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

        # Agents mostly use text; we expose a text-only pipeline via tokenizer
        tokenizer = processor.tokenizer
        text_generation_pipeline = TextGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
        )

        main_logger.info("[main] Qwen3-VL model and processor loaded successfully.")
    else:
        main_logger.info("[main] Loading generic CausalLM model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            device_map="auto",  # Auto-allocate to GPUs
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        text_generation_pipeline = TextGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
        )
        main_logger.info("[main] Model and tokenizer loaded successfully.")

    # ========== 3. Initialize Agents ==========
    main_logger.info(
        "[main] Initializing PerceptionAgent for retrieval component..."
    )
    perception_agent = GraphPerceptionAgent(
        data_file=args.data_file,
        llm_pipeline=text_generation_pipeline,
        max_new_tokens=args.max_tokens,
        top_percent=args.top_percent,
        sample_size=args.sample_size,
    )

    main_logger.info("[main] Initializing EnhancementAgent for generation component...")
    enhancement_agent = GraphEnhancementAgent(
        text_generation_pipeline=text_generation_pipeline,
        max_new_tokens=args.max_tokens,
    )

    main_logger.info("[main] Initializing EvaluationAgent for verification component...")
    evaluation_agent = GraphEvaluationAgent(
        text_generation_pipeline=text_generation_pipeline,
        max_new_tokens=args.max_tokens,
    )

    # ========== 4. Initialize ManagerAgent ==========
    main_logger.info(
        f"[main] Initializing ManagerAgent with enhancement_mode={args.enhancement_mode}..."
    )
    manager_agent = ManagerAgent(
        text_generation_pipeline=text_generation_pipeline,
        perception_agent=perception_agent,
        enhancement_agent=enhancement_agent,
        evaluation_agent=evaluation_agent,
        data_file=args.data_file,
        visualize_sampling=args.visualize_sampling,
        enhancement_mode=args.enhancement_mode,  # Can be None, auto-decide
        target_label=args.target_label,
        label_target_size=args.label_target_size,
    )

    # ========== 5. RAG-Based Iterative Enhancement Loop ==========
    main_logger.info(
        "[main] Starting GraphMaster RAG-based iterative enhancement loop..."
    )
    iteration = 0
    continue_flag = True
    iteration_history = []  # Store history for visualization
    current_data_str = None  # Will be loaded by manager_agent

    # Create enhanced data file
    original_data_file = args.data_file
    enhanced_data_file = original_data_file.replace(".json", "_enhanced.json")
    if not os.path.exists(enhanced_data_file):
        create_enhanced_file(original_data_file, enhanced_data_file)
        main_logger.info(
            f"[main] Enhanced file created: {enhanced_data_file} (copied from {original_data_file})"
        )
    else:
        main_logger.info(
            f"[main] Found existing enhanced file: {enhanced_data_file}. Continuing enhancement."
        )

    # Update data file for enhancement
    args.data_file = enhanced_data_file

    # Log baseline state
    main_logger.info("[main] Generating baseline environment report...")
    baseline_report = perception_agent.generate_environment_report(
        require_label_distribution=True
    )
    result_logger.info(
        f"********* BASELINE STATE *********\n{baseline_report}\n"
    )

    # Iterative RAG optimization loop
    while continue_flag and iteration < args.max_iterations:
        main_logger.info(f"\n[main] --- Iteration {iteration + 1} ---")
        main_logger.info(
            f"[main] Current enhancement mode: {manager_agent.enhancement_mode}"
        )

        # Execute one iteration of the RAG pipeline
        evaluated_data_str, continue_flag = manager_agent.run_manager_pipeline(
            early_stopping=args.early_stopping,
            current_iteration=iteration,
            current_data_str=current_data_str,
        )

        if not evaluated_data_str.strip():
            main_logger.info(
                "[main] No valid enhancement data obtained. Stopping iterations."
            )
            break

        # Clean and format the JSON data
        cleaned_data_str = reformat_json_str(evaluated_data_str)
        main_logger.info(
            f"[main] Cleaned JSON data at iteration {iteration + 1}"
        )

        # Store current iteration's result
        current_mode = manager_agent.enhancement_mode
        iteration_data = {
            "iteration": iteration + 1,
            "mode": current_mode,
            "enhanced_data": cleaned_data_str,
            "weights": {
                "semantic": manager_agent.lambda_sem,
                "structural": manager_agent.lambda_struct,
                "balance": manager_agent.lambda_bal,
            },
        }
        iteration_history.append(iteration_data)

        # Update the enhanced data file
        update_enhanced_data(enhanced_data_file, cleaned_data_str)

        # Log the result
        result_logger.info(
            f"**************** ITERATION {iteration+1} ****************"
        )
        result_logger.info(f"Mode: {current_mode}")
        result_logger.info(
            f"Adaptive Weights: Semantic={manager_agent.lambda_sem:.2f}, "
            + f"Structural={manager_agent.lambda_struct:.2f}, "
            + f"Balance={manager_agent.lambda_bal:.2f}"
        )
        result_logger.info(f"Enhanced Data: {cleaned_data_str}")

        # Update current_data_str for next iteration
        current_data_str = cleaned_data_str

        # Increment iteration counter
        iteration += 1

        if not continue_flag:
            main_logger.info(
                "[main] ManagerAgent indicates optimization convergence."
            )
            break

    # ========== 6. Generate Final Report and Visualizations ==========
    main_logger.info("[main] Generating final environment report...")
    final_report = perception_agent.generate_environment_report(
        require_label_distribution=True
    )

    # Visualize the enhancement process
    if args.visualize_sampling:
        generate_enhancement_visualization(
            iteration_history, baseline_report, final_report
        )

    # Log final state
    result_logger.info(
        f"********* FINAL STATE AFTER {iteration} ITERATIONS *********"
    )
    result_logger.info(
        f"Final Enhancement Mode: {manager_agent.enhancement_mode}"
    )
    result_logger.info(f"Final Report: {final_report}")

    main_logger.info(
        "[main] GraphMaster RAG-based enhancement pipeline completed successfully."
    )


def generate_enhancement_visualization(
    iteration_history, baseline_report, final_report
):
    """
    Generates visualizations of the enhancement process to show progress.
    """
    main_logger = logging.getLogger("main_logger")
    main_logger.info(
        "[main] Generating enhancement process visualization..."
    )

    try:
        # Extract data for visualization
        iterations = [data["iteration"] for data in iteration_history]
        modes = [data["mode"] for data in iteration_history]

        # Extract weights over iterations
        sem_weights = [data["weights"]["semantic"] for data in iteration_history]
        struct_weights = [data["weights"]["structural"] for data in iteration_history]
        bal_weights = [data["weights"]["balance"] for data in iteration_history]

        # Plot adaptive weights evolution
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, sem_weights, "r-", label="Semantic Weight")
        plt.plot(iterations, struct_weights, "g-", label="Structural Weight")
        plt.plot(iterations, bal_weights, "b-", label="Balance Weight")
        plt.xlabel("Iteration")
        plt.ylabel("Weight Value")
        plt.title("Evolution of Adaptive Weights in GraphMaster")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.savefig(
            "adaptive_weights_evolution.png", dpi=300, bbox_inches="tight"
        )

        # Parse baseline and final reports
        baseline_data = json.loads(baseline_report)
        final_data = json.loads(final_report)

        # Extract label distributions
        if "LabelDistribution" in baseline_data and "LabelDistribution" in final_data:
            # Plot label distribution change
            baseline_labels = baseline_data["LabelDistribution"]
            final_labels = final_data["LabelDistribution"]

            labels = sorted(
                list(set(baseline_labels.keys()) | set(final_labels.keys()))
            )
            baseline_counts = [
                int(baseline_labels.get(str(label), 0)) for label in labels
            ]
            final_counts = [
                int(final_labels.get(str(label), 0)) for label in labels
            ]

            plt.figure(figsize=(12, 6))
            x = np.arange(len(labels))
            width = 0.35

            plt.bar(x - width / 2, baseline_counts, width, label="Baseline")
            plt.bar(x + width / 2, final_counts, width, label="Final")

            plt.xlabel("Label")
            plt.ylabel("Count")
            plt.title("Label Distribution: Baseline vs Final")
            plt.xticks(x, labels)
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.savefig(
                "label_distribution_change.png", dpi=300, bbox_inches="tight"
            )

        main_logger.info("[main] Visualizations generated and saved.")
    except Exception as e:
        main_logger.error(
            f"[main] Error generating visualizations: {e}"
        )


def ensure_model_downloaded(model_name, cache_dir, token=None):
    """
    Checks if the specified model exists in cache_dir,
    downloads it if not. Supports using token for gated models.
    对于本地路径(例如 /apdcephfs/.../Qwen3-VL-8B-Instruct/)，直接跳过下载。
    """
    logger = logging.getLogger("main_logger")

    # If it looks like a local path (absolute or relative), skip HF download
    if os.path.isabs(model_name) or model_name.startswith("."):
        logger.info(
            f"[ensure_model_downloaded] Treating '{model_name}' as a local path. Skipping HF download."
        )
        return

    # Only for HF Hub model names (e.g., 'Qwen/QwQ-32B-Preview')
    model_path = os.path.join(
        cache_dir, "models--" + model_name.replace("/", "--")
    )
    if not os.path.exists(model_path):
        logger.info(
            f"[ensure_model_downloaded] Model not found in cache: {model_path}, downloading..."
        )
        try:
            # Prepare token if provided
            token_kwargs = {"token": token} if token else {}

            logger.info(
                f"[ensure_model_downloaded] Downloading tokenizer for {model_name}..."
            )
            AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                **token_kwargs,
            )

            logger.info(
                f"[ensure_model_downloaded] Downloading model for {model_name}..."
            )
            AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                **token_kwargs,
            )

            logger.info(
                "[ensure_model_downloaded] Model downloaded successfully."
            )
        except Exception as e:
            logger.error(
                f"[ensure_model_downloaded] Error downloading model: {e}"
            )

            # Special handling for authentication errors
            if (
                "token" in str(e).lower()
                or "permission" in str(e).lower()
                or "access" in str(e).lower()
            ):
                logger.error(
                    "\n[ensure_model_downloaded] AUTHENTICATION ERROR: This model requires Hugging Face authentication."
                )
                logger.error("[ensure_model_downloaded] Please:")
                logger.error(
                    "  1. Request access to this model on huggingface.co"
                )
                logger.error(
                    "  2. Run 'huggingface-cli login' in your terminal"
                )
                logger.error(
                    "  3. Or add --hf_token YOUR_TOKEN to your command line arguments\n"
                )
            raise
    else:
        logger.info(
            f"[ensure_model_downloaded] Model found in cache: {model_path}. Skipping download."
        )

def create_enhanced_file(data_file, enhanced_file):
    """
    Creates a copy of the data file and saves it as enhanced_file.
    """
    if not os.path.exists(data_file):
        logging.error(
            f"[create_enhanced_file] Error: {data_file} file does not exist!"
        )
        return

    with open(data_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    with open(enhanced_file, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def update_enhanced_data(enhanced_data_file, cleaned_data_str):
    """
    Updates the enhanced data file by APPENDING new nodes to existing data
    
    Args:
        enhanced_data_file: Path to the enhanced data JSON file
        cleaned_data_str: Cleaned JSON data as string (NEW nodes only)
    """
    main_logger = logging.getLogger("main_logger")
    
    if not cleaned_data_str or cleaned_data_str == "[]":
        main_logger.warning("[main] No cleaned data to update enhanced file.")
        return
    
    try:
        # Parse new nodes
        new_nodes = json.loads(cleaned_data_str)
        
        if not isinstance(new_nodes, list):
            main_logger.error("[main] Cleaned data is not a list")
            return
        
        if len(new_nodes) == 0:
            main_logger.warning("[main] New nodes list is empty")
            return
        
        # Load existing data from enhanced file
        existing_data = []
        if os.path.exists(enhanced_data_file):
            try:
                with open(enhanced_data_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = []
                main_logger.info(f"[main] Loaded {len(existing_data)} existing nodes from enhanced file")
            except (json.JSONDecodeError, IOError) as e:
                main_logger.error(f"[main] Error loading existing data: {e}")
                existing_data = []
        
        # Get existing node IDs to avoid duplicates
        existing_ids = {node.get('node_id') for node in existing_data if 'node_id' in node}
        
        # Filter out duplicate nodes
        unique_new_nodes = [node for node in new_nodes if node.get('node_id') not in existing_ids]
        
        if len(unique_new_nodes) < len(new_nodes):
            main_logger.warning(f"[main] Filtered out {len(new_nodes) - len(unique_new_nodes)} duplicate nodes")
        
        if len(unique_new_nodes) == 0:
            main_logger.warning("[main] No unique new nodes to add")
            return
        
        # Reassign node IDs to ensure sequential ordering
        # Find max numeric ID in existing data
        max_id = 0
        for node in existing_data:
            node_id = node.get('node_id', '')
            # Try to extract numeric part
            if isinstance(node_id, int):
                max_id = max(max_id, node_id)
            elif isinstance(node_id, str):
                # Extract numbers from string (e.g., "new_node_123" -> 123)
                numbers = re.findall(r'\d+', node_id)
                if numbers:
                    max_id = max(max_id, int(numbers[-1]))
        
        # Renumber new nodes starting from max_id + 1
        for i, node in enumerate(unique_new_nodes, start=1):
            old_id = node.get('node_id', '')
            new_id = f"new_node_{max_id + i}"
            node['node_id'] = new_id
            
            # Update neighbor references if they point to other new nodes
            if 'neighbors' in node:
                updated_neighbors = []
                for neighbor_id in node['neighbors']:
                    # Keep existing node references as is
                    # Only update if neighbor is also a new node
                    found = False
                    for j, other_node in enumerate(unique_new_nodes):
                        if other_node.get('node_id') == neighbor_id:
                            updated_neighbors.append(f"new_node_{max_id + j + 1}")
                            found = True
                            break
                    if not found:
                        updated_neighbors.append(neighbor_id)
                node['neighbors'] = updated_neighbors
        
        # Combine existing and new data
        combined_data = existing_data + unique_new_nodes
        
        # Write back to file
        with open(enhanced_data_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)
        
        main_logger.info(f"[main] Successfully updated enhanced file:")
        main_logger.info(f"[main]   - Existing nodes: {len(existing_data)}")
        main_logger.info(f"[main]   - New nodes added: {len(unique_new_nodes)}")
        main_logger.info(f"[main]   - Total nodes: {len(combined_data)}")
        
    except json.JSONDecodeError as e:
        main_logger.error(f"[main] Error parsing cleaned data: {e}")
    except Exception as e:
        main_logger.error(f"[main] Error updating enhanced data file: {e}")
        import traceback
        main_logger.error(f"[main] Traceback:\n{traceback.format_exc()}")    
        """
    Updates the enhanced data file with cleaned data
    
    Args:
        enhanced_data_file: Path to the enhanced data JSON file
        cleaned_data_str: Cleaned JSON data as string
    """
    main_logger = logging.getLogger("main_logger")
    
    if not cleaned_data_str:
        main_logger.warning("[main] No cleaned data to update enhanced file.")
        return
    
    try:
        # Parse the cleaned data
        cleaned_data = json.loads(cleaned_data_str)
        
        if not isinstance(cleaned_data, list):
            main_logger.error("[main] Cleaned data is not a list")
            return
        
        # Write to enhanced data file
        with open(enhanced_data_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        
        main_logger.info(f"[main] Updated enhanced data file with {len(cleaned_data)} nodes")
        
    except json.JSONDecodeError as e:
        main_logger.error(f"[main] Error parsing cleaned data: {e}")
    except Exception as e:
        main_logger.error(f"[main] Error updating enhanced data file: {e}")
        import traceback
        main_logger.error(f"[main] Traceback:\n{traceback.format_exc()}")


def reformat_json_str(content):
    """
    Converts messy JSON string to standard format.
    """
    objs = fix_messy_json(content)
    return json.dumps(objs, ensure_ascii=False, indent=4)


def fix_messy_json(content):
    """
    从 LLM 输出中提取合法的 JSON 数组
    """
    logger = logging.getLogger("main_logger")

    if not isinstance(content, str) or not content:
        logger.error("[fix_messy_json] Empty or invalid content")
        return []

    text = content.strip()

    # 1. 移除 markdown 代码块
    if text.startswith("```"):
        lines = text.split('\n')
        # 移除第一行 ```json 或 ```
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        # 移除最后的 ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = '\n'.join(lines).strip()

    # 2. 查找 flag 后的内容
    flag = "here are the generated datasets:"
    idx = text.lower().find(flag.lower())
    if idx != -1:
        text = text[idx + len(flag):].strip()

    # 3. 提取 JSON 数组
    start = text.find("[")
    end = text.rfind("]")
    
    if start == -1 or end == -1 or end <= start:
        logger.error("[fix_messy_json] No valid JSON array found")
        logger.error(f"Content (first 300 chars): {content[:300]}")
        return []

    json_str = text[start:end + 1]

    # 4. 尝试解析
    try:
        data = json.loads(json_str)
        if not isinstance(data, list):
            logger.error(f"[fix_messy_json] Parsed data is not a list: {type(data)}")
            return []
        return data
    except json.JSONDecodeError as e:
        logger.error(f"[fix_messy_json] JSON parsing failed: {e}")
        logger.error(f"Failed JSON (first 500 chars): {json_str[:500]}")
        return []


def setup_logging():
    os.makedirs("./log", exist_ok=True)
    os.makedirs("./log/log_main", exist_ok=True)
    os.makedirs("./log/log_result", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    main_logger = logging.getLogger("main_logger")
    main_logger.setLevel(logging.INFO)
    main_log_file = f"./log/log_main/{timestamp}.log"
    main_handler = logging.FileHandler(
        main_log_file, mode="w", encoding="utf-8"
    )
    main_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    main_logger.addHandler(main_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    main_logger.addHandler(console_handler)  # Add terminal output

    result_logger = logging.getLogger("result_logger")
    result_logger.setLevel(logging.INFO)
    result_log_file = f"./log/log_result/enhancement_{timestamp}.log"
    result_handler = logging.FileHandler(
        result_log_file, mode="w", encoding="utf-8"
    )
    result_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    result_console_handler = logging.StreamHandler()
    result_console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    result_logger.addHandler(result_handler)
    result_logger.addHandler(result_console_handler)  # Add terminal output

    return main_logger, result_logger

if __name__ == "__main__":
    main()
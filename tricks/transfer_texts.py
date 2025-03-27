import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

# =========================
# 1. 设置模型环境变量与模型名称
# =========================
cache_dir = "/home/ai/EnjunDu/model"  # 模型缓存目录
os.environ["HF_HOME"] = cache_dir
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"  # 目标模型名称

# =========================
# 2. 读取 cora.json
# =========================
with open('./data/cora.json', 'r', encoding='utf-8') as file:
    dataset = json.load(file)  # 读取 JSON 数据，列表形式存储所有节点

# =========================
# 3. 初始化模型和 Tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    device_map="auto",       # 自动分配到 GPU
    torch_dtype=torch.float16 # 适用于大模型的半精度计算
)
text_generation = TextGenerationPipeline(model=model, tokenizer=tokenizer)

# =========================
# 4. 生成文本压缩 Prompt（英文）
# =========================
def generate_text_summarization_prompt(text):
    """
    生成 LLM 需要的 Prompt，要求模型将 text 压缩至 35 个单词以内。
    """
    prompt = f"""
You are an AI assistant specializing in text summarization.
Your task is to compress the following academic text while preserving its core meaning.
Ensure that the summary is concise and contains no more than 30 words.
Please think step by step.

Original text:
{text}

Summarized text (30 words or fewer):
"""
    return prompt

# =========================
# 5. 处理单个节点文本（只返回模型输出的摘要，不包含 prompt 部分）
# =========================
def compress_text(text):
    """
    使用 DeepSeek-R1 32B 对 text 进行压缩，限制在 35 个单词以内。
    提取生成文本中 "Summarized text (35 words or fewer):" 标记后的内容作为摘要。
    """
    prompt = generate_text_summarization_prompt(text)

    # 调用 LLM 生成压缩文本
    output = text_generation(
        prompt,
        max_new_tokens=45,  # 生成最多 50 个 token（大约35个单词）
        do_sample=False,     # 关闭随机采样，保证一致性
        temperature=0.0      # 降低创造性，确保摘要精确
    )

    # 获取生成的完整文本
    generated_text = output[0]["generated_text"].strip()

    # 提取模型输出中摘要部分：寻找 "Summarized text (35 words or fewer):" 后的内容
    marker = "Summarized text (30 words or fewer):"
    if marker in generated_text:
        # 分割后取最后一部分作为摘要
        summary = generated_text.split(marker)[-1].strip()
    else:
        # 如果找不到 marker，则直接返回全部内容（不建议出现这种情况）
        summary = generated_text

    # 过滤掉模型可能生成的 "</think>" 标记及其前后的空白字符
    summary = summary.replace("</think>", "").strip()

    return summary

# =========================
# 6. 处理整个数据集
# =========================
enhanced_dataset = []  # 存储处理后的数据

for i, node in enumerate(dataset):
    print(f"Processing node {i + 1}/{len(dataset)} (ID: {node['node_id']})...")

    # 生成压缩后的 text（只保留模型输出的摘要部分）
    compressed_text = compress_text(node["text"])

    # 构造增强后的节点，仅保存模型生成的摘要
    enhanced_node = {
        "node_id": node["node_id"],
        "label": node["label"],
        "text": compressed_text,  # 仅保存模型生成的摘要
        "neighbors": node["neighbors"],
        "mask": node["mask"]
    }

    enhanced_dataset.append(enhanced_node)

# =========================
# 7. 保存至 cora_enhanced.json
# =========================
output_path = "./data/cora_enhanced_30.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(enhanced_dataset, f, ensure_ascii=False, indent=2)

print(f"Processing complete! Enhanced dataset saved to {output_path}")
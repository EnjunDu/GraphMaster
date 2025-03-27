import pickle
import os
import torch
import logging
import argparse
import json
import scipy.sparse as sp

from tqdm import tqdm
from models.GCN_llm import GCN
from models.GAT_llm import GAT
from models.JKNet_llm import JKNet
from models.GSAGE_llm import GraphSAGE

# ====== 使用 Sentence-BERT ====== #
from sentence_transformers import SentenceTransformer

def setup_logging(log_filename):
    logging.basicConfig(
        filename=log_filename,
        filemode='w',
        level=logging.INFO
    )
    logging.getLogger().addHandler(logging.StreamHandler())

def get_text_embedding(text, model, device='cpu'):
    embedding = model.encode(text, convert_to_tensor=True, show_progress_bar=False)
    return embedding.cpu()

def train_gnn_model(model_name, adj_orig, features, labels, tvt_nids, args):
    """
    Train a specific GNN model and return its performance metrics
    """
    logging.info(f"Training {model_name} model...")
    
    if model_name == 'GCN':
        model = GCN(
            adj=adj_orig,
            adj_eval=adj_orig,
            features=features,
            labels=labels,
            tvt_nids=tvt_nids,
            cuda=args.gpu,
            hidden_size=args.hidden_size,
            n_layers=args.n_layers,
            epochs=args.epochs,
            seed=args.seed,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            print_progress=True,
            dropedge=args.dropedge
        )
    elif model_name == 'GAT':
        model = GAT(
            adj=adj_orig,
            adj_eval=adj_orig,
            features=features,
            labels=labels,
            tvt_nids=tvt_nids,
            cuda=args.gpu,
            hidden_size=args.hidden_size,
            n_layers=args.n_layers,
            epochs=args.epochs,
            seed=args.seed,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            print_progress=True,
            dropedge=args.dropedge,
            attn_drop=args.dropout,
            negative_slope=0.2
        )
    elif model_name == 'JKNet':
        model = JKNet(
            adj=adj_orig,
            adj_eval=adj_orig,
            features=features,
            labels=labels,
            tvt_nids=tvt_nids,
            cuda=args.gpu,
            hidden_size=args.hidden_size,
            n_layers=args.n_layers,
            epochs=args.epochs,
            seed=args.seed,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            print_progress=True,
            dropedge=args.dropedge
        )
    elif model_name == 'GraphSAGE':
        model = GraphSAGE(
            adj=adj_orig,
            adj_eval=adj_orig,
            features=features,
            labels=labels,
            tvt_nids=tvt_nids,
            cuda=args.gpu,
            hidden_size=args.hidden_size,
            n_layers=args.n_layers,
            epochs=args.epochs,
            seed=args.seed,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            print_progress=True,
            dropedge=args.dropedge
        )
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    
    # Train the model and get results
    best_test_acc, best_test_prec, best_test_rec, best_test_f1, best_epoch = model.fit()
    
    logging.info(f"[{model_name}] best_test_acc: {best_test_acc:.4f}, best_test_prec: {best_test_prec:.4f}, "
                f"best_test_rec: {best_test_rec:.4f}, best_test_f1: {best_test_f1:.4f}, best_epoch: {best_epoch}")
    
    return {
        'model': model_name,
        'acc': best_test_acc,
        'prec': best_test_prec,
        'rec': best_test_rec,
        'f1': best_test_f1,
        'epoch': best_epoch
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train multiple GNN models on a dataset')
    parser.add_argument('--dataset', type=str, default='cora', help="数据集名称，例如：cora、citeseer、pubmed")
    parser.add_argument('--gpu', type=int, default=1, help="GPU编号，-1表示使用CPU")
    parser.add_argument('--v',action='store_true', help="Vertification the enhancement data")
    parser.add_argument('--hidden_size', type=int, default=128, help="隐藏层大小")
    parser.add_argument('--n_layers', type=int, default=2, help="GNN的层数")
    parser.add_argument('--epochs', type=int, default=1000, help="训练的轮数")
    parser.add_argument('--seed', type=int, default=42, help="随机种子")
    parser.add_argument('--lr', type=float, default=1e-2, help="学习率")
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="权重衰减系数")
    parser.add_argument('--dropout', type=float, default=0.35, help="dropout比例")
    parser.add_argument('--dropedge', type=float, default=0.0, help="边的丢弃比例（仅部分模型支持）")
    parser.add_argument('--log_file', type=str, default='all_models', help="日志文件名")
    args = parser.parse_args()

    # ============ 0. 设置日志文件 ============ #
    if not os.path.exists('./log'):
        os.makedirs('./log')
    setup_logging(f'./log/{args.log_file}_{args.dataset}.log')

    # ============ 0.1 设置设备 ============ #
    if args.gpu == -1 or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        device = torch.device('cuda:3')  # 保持与原代码一致，使用固定的 GPU 3

    # ============ 1. 加载节点数据 (JSON) 和生成特征 (Sentence-BERT) ============ #
    if args.v:
        json_path = f"../data/{args.dataset}_enhanced.json"
    else:
        json_path = f"../data/{args.dataset}.json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        node_data = json.load(f)

    # 指定 Sentence-BERT 模型存储路径
    default_model_path = "/home/ai/EnjunDu/model/sentence-bert"
    alternative_model_path = "/home/daihengwei/EnjunDu/model/sentence-bert"
    if os.path.exists(default_model_path):
        model_path = default_model_path
    else:
        model_path = alternative_model_path
    fallback_model_name = "sentence-transformers/all-mpnet-base-v2"  # 如果本地加载失败时回退的模型

    # 尝试加载本地 Sentence-BERT 模型，否则在线下载并保存
    try:
        logging.info(f"Trying to load local Sentence-BERT model from: {model_path}")
        llm_model = SentenceTransformer(model_path)
    except Exception as e:
        logging.warning(f"Failed to load local Sentence-BERT model from {model_path}. Error: {e}")
        logging.info(f"Loading fallback model {fallback_model_name} from Hugging Face.")
        llm_model = SentenceTransformer(fallback_model_name)
        # 将下载的模型保存到本地 model_path
        llm_model.save(model_path)

    # 根据 node_id 排序，确保后续处理（特征、邻接等）与节点顺序一一对应
    node_data_sorted = sorted([x for x in node_data if 'node_id' in x], key=lambda x: x['node_id'])
    num_nodes = len(node_data_sorted)

    # ============ 1.1 生成节点文本向量特征 ============ #
    features_list = []
    for item in tqdm(node_data_sorted, desc="Generating features..."):
        text = item['text']
        embedding = get_text_embedding(text, llm_model, device=device)
        features_list.append(embedding)
    # 拼接成 [N, embed_dim] 的张量
    features = torch.stack(features_list, dim=0)

    # ============ 2. 从 JSON 中生成 tvt_nids, adj_orig, labels ============ #
    train_nids = []
    val_nids = []
    test_nids = []
    labels_list = [0] * num_nodes

    adjacency = sp.lil_matrix((num_nodes, num_nodes))

    for i, item in tqdm(enumerate(node_data_sorted), desc="Processing nodes..."):
        node_id = item["node_id"]
        mask = item["mask"]
        label = item["label"]
        neighbors = item["neighbors"]

        # 根据 mask 分配 train/val/test
        if mask == "Train":
            train_nids.append(node_id)
        elif mask == "Validation":
            val_nids.append(node_id)
        elif mask == "Test":
            test_nids.append(node_id)
        else:
            logging.warning(f"Node {node_id} has an unknown mask type: {mask}")

        labels_list[node_id] = label
        valid_node_ids = set(item['node_id'] for item in node_data_sorted)

        for nbr in neighbors:
            if not isinstance(nbr, int):
                raise TypeError(f"Neighbor {nbr} is not an integer!")
            if nbr not in valid_node_ids:
                raise ValueError(f"Neighbor {nbr} for node {node_id} is not in node_id list!")

            adjacency[node_id, nbr] = 1
            adjacency[nbr, node_id] = 1

    adj_orig = adjacency.tocsr()

    # ============ 2.1 自动重映射标签以确保标签范围从0开始 ============
    unique_original_labels = sorted(set(labels_list))
    logging.info(f"Unique labels (original): {unique_original_labels}")

    # 建立映射 old_label -> 新的 label 索引
    label_to_idx = {old_label: idx for idx, old_label in enumerate(unique_original_labels)}
    # 将 labels_list 中所有标签转换成新的索引
    for i in range(len(labels_list)):
        old_label = labels_list[i]
        labels_list[i] = label_to_idx[old_label]

    # 检查映射后标签的最小值和最大值
    min_label = min(labels_list)
    max_label = max(labels_list)
    logging.info(f"Remapped label range: min={min_label}, max={max_label}")
    # 如果仍然不满足 0 <= label < n_class，就报错
    if min_label < 0 or max_label >= len(unique_original_labels):
        raise ValueError(
            "Some labels are out of valid range after re-mapping: "
            f"[{min_label}, {max_label}] vs [0, {len(unique_original_labels)-1}]"
        )

    labels = torch.tensor(labels_list, dtype=torch.long)
    tvt_nids = {
        'train': train_nids,
        'val': val_nids,
        'test': test_nids
    }

    # ============ 3. 训练所有四个 GNN 模型 ============ #
    all_models = ['GCN', 'GAT', 'JKNet', 'GraphSAGE']
    results = []

    for model_name in all_models:
        result = train_gnn_model(model_name, adj_orig, features, labels, tvt_nids, args)
        results.append(result)

    # ============ 4. 找出每个指标的最大值 ============ #
    best_acc = max(results, key=lambda x: x['acc'])
    best_prec = max(results, key=lambda x: x['prec'])
    best_rec = max(results, key=lambda x: x['rec'])
    best_f1 = max(results, key=lambda x: x['f1'])

    # ============ 5. 输出最终结果 ============ #
    logging.info("=" * 50)
    logging.info(f"args: {args}")
    logging.info("Best metrics across all models:")
    logging.info(f"Best Accuracy: {best_acc['acc']:.4f} (from {best_acc['model']})")
    logging.info(f"Best Precision: {best_prec['prec']:.4f} (from {best_prec['model']})")
    logging.info(f"Best Recall: {best_rec['rec']:.4f} (from {best_rec['model']})")
    logging.info(f"Best F1 Score: {best_f1['f1']:.4f} (from {best_f1['model']})")
    logging.info("=" * 50)
    
    # 输出所有模型的结果总表
    logging.info("All models results summary:")
    header = "| {:^10} | {:^10} | {:^10} | {:^10} | {:^10} |".format("Model", "Accuracy", "Precision", "Recall", "F1 Score")
    separator = "-" * len(header)
    logging.info(separator)
    logging.info(header)
    logging.info(separator)
    print(f"results for {args.dataset}:")
    for result in results:
        logging.info("| {:^10} | {:.4f}     | {:.4f}     | {:.4f}     | {:.4f}     |".format(
            result['model'], result['acc'], result['prec'], result['rec'], result['f1']))
        
    logging.info(separator)
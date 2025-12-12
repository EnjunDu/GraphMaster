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

# ====== Using Sentence-BERT ====== #
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='single')
    parser.add_argument('--dataset', type=str, default='cora', help="The name of the dataset, for example:cora、citeseer、pubmed")
    parser.add_argument('--gnn', type=str, default='GCN', help="The selected GNN model, for example: GCN, GAT, JKNet, GraphSAGE")
    parser.add_argument('--gpu', type=int, default=1, help="GPU number, -1 means using CPU")
    parser.add_argument('--v',action='store_true', help="Vertification the enhancement data")
    parser.add_argument('--hidden_size', type=int, default=128, help="Hidden layer size")
    parser.add_argument('--n_layers', type=int, default=2, help="Number of layers of GNN")
    parser.add_argument('--epochs', type=int, default=1000, help="Number of training rounds")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="Weight attenuation coefficient")
    parser.add_argument('--dropout', type=float, default=0.35, help="dropout ratio")
    parser.add_argument('--dropedge', type=float, default=0.0, help="Edge discard ratio (supported only by some models)")
    parser.add_argument('--log_file', type=str, default='original', help="Log file name")
    args = parser.parse_args()

    # ============ 0. Set up log files ============ #
    if not os.path.exists('./log'):
        os.makedirs('./log')
    setup_logging(f'./log/{args.log_file}_{args.gnn}_{args.dataset}.log')

    # ============ 0.1 Set up the device ============ #
    if args.gpu == -1 or not torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        device = torch.device('cuda:3')  # cuda:3 is fixed here and can be modified as needed
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        device = torch.device('cuda:3')

    # ============ 1. Loading node data (JSON) and generating features (Sentence-BERT) ============ #
    if args.v:
        json_path = f"../data/{args.dataset}_enhanced.json"
    else:
        json_path = f"../data/{args.dataset}.json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        node_data = json.load(f)

    # Specify the Sentence-BERT model storage path
    # Specify the Sentence-BERT model storage path: first check if the default path exists, otherwise use the alternate path
    default_model_path = "/home/ai/EnjunDu/model/sentence-bert"
    alternative_model_path = "/home/daihengwei/EnjunDu/model/sentence-bert"
    if os.path.exists(default_model_path):
        model_path = default_model_path
    else:
        model_path = alternative_model_path
    fallback_model_name = "sentence-transformers/all-mpnet-base-v2"  # Fallback model if local loading fails

    # Try to load the local Sentence-BERT model, otherwise download and save it online
    try:
        logging.info(f"Trying to load local Sentence-BERT model from: {model_path}")
        llm_model = SentenceTransformer(model_path)
    except Exception as e:
        logging.warning(f"Failed to load local Sentence-BERT model from {model_path}. Error: {e}")
        logging.info(f"Loading fallback model {fallback_model_name} from Hugging Face.")
        llm_model = SentenceTransformer(fallback_model_name)
        # Save the downloaded model to local model_path
        llm_model.save(model_path)

    # Sort by node_id to ensure that subsequent processing (features, adjacency, etc.) corresponds to the node order.
    node_data_sorted = sorted([x for x in node_data if 'node_id' in x], key=lambda x: x['node_id'])
    num_nodes = len(node_data_sorted)

    # ============ 1.1 Generate node text vector features ============ #
    features_list = []
    for item in tqdm(node_data_sorted, desc="Generating features..."):
        text = item['text']
        embedding = get_text_embedding(text, llm_model, device=device)
        features_list.append(embedding)
    # Concatenate into a tensor of [N, embed_dim]
    features = torch.stack(features_list, dim=0)

    # ============ 2. Generate tvt_nids, adj_orig, labels from JSON ============ #
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

        # Assign train/val/test according to mask
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

    # [MODIFIED] ============ 2.1 Automatically remap labels to ensure label ranges start at 0 ============
    unique_original_labels = sorted(set(labels_list))
    logging.info(f"Unique labels (original): {unique_original_labels}")

    # Create a mapping old_label -> new label index
    label_to_idx = {old_label: idx for idx, old_label in enumerate(unique_original_labels)}
    # Convert all labels in labels_list to new indices
    for i in range(len(labels_list)):
        old_label = labels_list[i]
        labels_list[i] = label_to_idx[old_label]

    # Check the minimum and maximum values ​​of the mapped labels
    min_label = min(labels_list)
    max_label = max(labels_list)
    logging.info(f"Remapped label range: min={min_label}, max={max_label}")
    # If 0 <= label < n_class is still not satisfied, an error is reported
    if min_label < 0 or max_label >= len(unique_original_labels):
        raise ValueError(
            "Some labels are out of valid range after re-mapping: "
            f"[{min_label}, {max_label}] vs [0, {len(unique_original_labels)-1}]"
        )
    # [MODIFIED] ============ 2.1 END ============ #

    labels = torch.tensor(labels_list, dtype=torch.long)
    tvt_nids = {
        'train': train_nids,
        'val': val_nids,
        'test': test_nids
    }

    # ============ 3. Instantiate according to the selected GNN model ============ #
    if args.gnn == 'GCN':
        model = GCN(
            adj=adj_orig,
            adj_eval=adj_orig,  # If the evaluation also uses the same adjacency matrix
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
    elif args.gnn == 'GAT':
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
    elif args.gnn == 'JKNet':
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
    elif args.gnn == 'GraphSAGE':
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
        raise ValueError(f"Unsupported model type: {args.gnn}")

    # ============ 4. Train the model and output the final result ============ #
    best_test_acc, best_test_prec, best_test_rec, best_test_f1, best_epoch = model.fit()
    logging.info(f"args: {args}")
    logging.info(f"[Final] best_test_acc: {best_test_acc:.4f}, best_test_prec: {best_test_prec:.4f}, "
                 f"best_test_rec: {best_test_rec:.4f}, best_test_f1: {best_test_f1:.4f}, best_epoch: {best_epoch}")

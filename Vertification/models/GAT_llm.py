import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

try:
    from torch_scatter import scatter_add
except ImportError:
    scatter_add = None

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def adjacency_to_edge_index(adj):
    """
    Convert a scipy sparse adjacency matrix to edge_index format: shape [2, E].
    """
    if not isinstance(adj, sp.coo_matrix):
        adj = sp.coo_matrix(adj)
    row = torch.from_numpy(adj.row).long()
    col = torch.from_numpy(adj.col).long()
    return torch.stack([row, col], dim=0)


class MyGraphAttentionLayer(nn.Module):
    """
    Single GAT layer with multi-head attention.
    """
    def __init__(
        self,
        in_features,
        out_features,
        num_heads=1,
        dropout=0.2,
        attn_dropout=0.2,
        alpha=0.2,
        concat=True,
        activation=None
    ):
        super(MyGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.activation = activation
        
        self.W = nn.Parameter(torch.Tensor(in_features, out_features * num_heads))
        self.a_src = nn.Parameter(torch.Tensor(num_heads, out_features))
        self.a_dst = nn.Parameter(torch.Tensor(num_heads, out_features))
        
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W, gain=1.0)
        nn.init.xavier_uniform_(self.a_src, gain=1.0)
        nn.init.xavier_uniform_(self.a_dst, gain=1.0)

    def forward(self, x, edge_index):
        N = x.size(0)
        x = self.dropout(x)
        # [N, in_features] -> [N, num_heads * out_features]
        x = torch.mm(x, self.W)
        # [N, num_heads, out_features]
        x = x.view(N, self.num_heads, self.out_features)

        row = edge_index[0]
        col = edge_index[1]

        # Compute attention scores
        alpha_src = (x[row] * self.a_src).sum(dim=-1)  # [E, num_heads]
        alpha_dst = (x[col] * self.a_dst).sum(dim=-1)  # [E, num_heads]
        alpha = self.leakyrelu(alpha_src + alpha_dst)

        # Group-wise softmax
        alpha = self.edge_softmax(alpha, row, N)
        alpha = self.attn_dropout(alpha)

        # Message passing
        out = self.message_passing(x, alpha, edge_index)

        if self.concat:
            out = out.view(N, self.num_heads * self.out_features)
        else:
            out = out.mean(dim=1)

        if self.activation is not None:
            out = self.activation(out)

        return out

    def edge_softmax(self, alpha, row, N):
        """
        Group softmax on 'row'. shape: alpha=[E, num_heads].
        """
        if scatter_add is None:
            # fallback
            alpha_out = torch.zeros_like(alpha)
            alpha_np = alpha.detach().cpu().numpy()
            row_np = row.detach().cpu().numpy()
            for i in range(N):
                mask = (row_np == i)
                if not np.any(mask):
                    continue
                sub_alpha = alpha_np[mask]
                sub_alpha = sub_alpha - np.max(sub_alpha, axis=0, keepdims=True)
                sub_alpha = np.exp(sub_alpha)
                sub_sum = np.sum(sub_alpha, axis=0, keepdims=True)
                sub_alpha = sub_alpha / (sub_sum + 1e-9)
                alpha_out[mask] = torch.from_numpy(sub_alpha).to(alpha.device)
            return alpha_out
        else:
            alpha = alpha - alpha.max(dim=0, keepdim=True)[0]
            alpha = alpha.exp()
            alpha_sum = scatter_add(alpha, row, dim=0, dim_size=N)
            return alpha / (alpha_sum[row] + 1e-9)

    def message_passing(self, x, alpha, edge_index):
        row = edge_index[0]
        col = edge_index[1]
        out = torch.zeros_like(x)
        if scatter_add is None:
            E = alpha.size(0)
            for e in range(E):
                i = row[e].item()
                j = col[e].item()
                out[i] += alpha[e].unsqueeze(-1) * x[j]
        else:
            agg = alpha.unsqueeze(-1) * x[col]
            out = scatter_add(agg, row, dim=0, dim_size=x.size(0))
        return out


class GAT_model(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, heads, dropout, attn_drop, negative_slope):
        super(GAT_model, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(
            MyGraphAttentionLayer(
                in_feats, n_hidden, heads[0],
                dropout=dropout, attn_dropout=attn_drop,
                alpha=negative_slope, concat=True, activation=activation
            )
        )
        # Hidden layers
        for i in range(1, n_layers):
            self.layers.append(
                MyGraphAttentionLayer(
                    n_hidden * heads[i-1], n_hidden, heads[i],
                    dropout=dropout, attn_dropout=attn_drop,
                    alpha=negative_slope, concat=True, activation=activation
                )
            )
        # Output layer (no concat)
        self.layers.append(
            MyGraphAttentionLayer(
                n_hidden * heads[-2], n_classes, heads[-1],
                dropout=dropout, attn_dropout=attn_drop,
                alpha=negative_slope, concat=False, activation=None
            )
        )

    def forward(self, edge_index, features):
        h = features
        for layer in self.layers[:-1]:
            h = layer(h, edge_index)
        return self.layers[-1](h, edge_index)


class GAT(object):
    def __init__(
        self,
        adj,
        adj_eval,
        features,
        labels,
        tvt_nids,
        cuda=-1,
        hidden_size=64,
        n_layers=2,
        epochs=1000,
        seed=42,
        lr=0.01,
        weight_decay=5e-4,
        dropout=0.25,
        print_progress=True,
        attn_drop=0.25,
        negative_slope=0.2,
        dropedge=0.0
    ):
        if cuda >= 0 and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{cuda%8}')
            torch.cuda.manual_seed(seed)
        else:
            self.device = torch.device('cpu')
        torch.manual_seed(seed)

        self.features = features.float().to(self.device)
        self.labels = labels.to(self.device)
        self.train_nid = tvt_nids['train']
        self.val_nid = tvt_nids['val']
        self.test_nid = tvt_nids['test']
        self.edge_index = adjacency_to_edge_index(adj).to(self.device)
        self.edge_index_eval = adjacency_to_edge_index(adj_eval).to(self.device)
        self.n_class = labels.max().item() + 1

        # define heads
        heads = [8] * n_layers + [1]
        self.model = GAT_model(
            in_feats=self.features.size(1),
            n_hidden=hidden_size,
            n_classes=self.n_class,
            n_layers=n_layers,
            activation=F.elu,
            heads=heads,
            dropout=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.epochs = epochs
        self.print_progress = print_progress
        self.dropedge = dropedge  # not used here, but consistent with other models

    def fit(self):
        best_val_f1 = 0.0
        best_test_acc, best_test_prec, best_test_rec, best_test_f1 = 0, 0, 0, 0
        best_epoch = 0

        for epoch in range(self.epochs):
            self.model.train()
            logits = self.model(self.edge_index, self.features)
            loss = F.cross_entropy(logits[self.train_nid], self.labels[self.train_nid])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                logits_eval = self.model(self.edge_index_eval, self.features)
            # Evaluate on validation set
            val_f1 = self.compute_f1(logits_eval[self.val_nid], self.labels[self.val_nid])

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                test_acc, test_prec, test_rec, test_f1 = self.eval_node_cls(logits_eval[self.test_nid], self.labels[self.test_nid])
                best_test_acc, best_test_prec, best_test_rec, best_test_f1 = test_acc, test_prec, test_rec, test_f1

            print(f"Epoch [{epoch+1}/{self.epochs}] loss={loss.item():.4f}, val_f1={val_f1:.4f}")
        
        if self.print_progress:
            print("Final test results:")
            print(f"  Acc: {best_test_acc:.4f}, Prec: {best_test_prec:.4f}, Rec: {best_test_rec:.4f}, F1: {best_test_f1:.4f}")
        return best_test_acc, best_test_prec, best_test_rec, best_test_f1, best_epoch

    def compute_f1(self, logits, labels):
        preds = logits.argmax(dim=1).cpu().numpy()
        labels = labels.cpu().numpy()
        return f1_score(labels, preds, average='macro')

    def eval_node_cls(self, logits, labels):
        preds = logits.argmax(dim=1).cpu().numpy()
        labels = labels.cpu().numpy()
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, average='macro', zero_division=0)
        rec = recall_score(labels, preds, average='macro', zero_division=0)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)
        return acc, prec, rec, f1

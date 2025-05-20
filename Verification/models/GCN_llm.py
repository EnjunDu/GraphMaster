import gc
import math
import time
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# =======================
# 1. The overall class of the model (not using DGL)
# =======================
class GCN(object):
    def __init__(self,
                 adj,
                 adj_eval,
                 features,
                 labels,
                 tvt_nids,
                 cuda=-1,
                 hidden_size=128,
                 n_layers=1,
                 epochs=200,
                 seed=-1,
                 lr=1e-2,
                 weight_decay=5e-4,
                 dropout=0.5,
                 print_progress=True,
                 dropedge=0):
        """
        The parameters are basically the same as before, except that the dependency on DGLGraph is removed, and the logic of L1 normalization for 1433/3703 dimensional features is removed.
        """
        self.t = time.time()               # Recording start time
        self.lr = lr                       # learning rate
        self.weight_decay = weight_decay   # L2 regularization coefficient
        self.epochs = epochs               # Number of training rounds
        self.print_progress = print_progress
        self.dropedge = dropedge           # edge drop rate

        # Configure compute devices (CPU/GPU)
        if not torch.cuda.is_available():
            cuda = -1
        self.device = torch.device(f'cuda:{cuda%8}' if cuda >= 0 else 'cpu')

        # Set random seed
        if seed > 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Data loading and preprocessing
        self.load_data(adj, adj_eval, features, labels, tvt_nids)

        # Initialize model
        self.model = GCN_model(
            in_feats=self.features.size(1),
            n_hidden=hidden_size,
            n_classes=self.n_class,
            n_layers=n_layers,
            activation=F.relu,
            dropout=dropout
        )
        self.model.to(self.device)

    def load_data(self, adj, adj_eval, features, labels, tvt_nids):
        """
        Without DGL, manually process the adjacency matrix and save it as a sparse PyTorch matrix (for training and evaluation)
        """

        # 1. Processing features (already vectors obtained from LLM, no more normalization)
        if isinstance(features, torch.FloatTensor) or isinstance(features, torch.cuda.FloatTensor):
            self.features = features
        else:
            self.features = torch.FloatTensor(features)

        # 2. Handle tags
        if len(labels.shape) == 2:
            labels = torch.FloatTensor(labels)  # multiple tags
        else:
            labels = torch.LongTensor(labels)   # single label
        self.labels = labels

        # 3. Determine the number of categories
        if len(self.labels.size()) == 1:
            self.n_class = len(torch.unique(self.labels))  # Single label: number of categories
        else:
            self.n_class = labels.size(1)                  # Multi-label: vector length

        # 4. Split into training/validation/test sets
        self.train_nid = tvt_nids['train']
        self.val_nid = tvt_nids['val']
        self.test_nid = tvt_nids['test']

        # 5. Constructing a sparse matrix for training
        assert sp.issparse(adj)
        adj = sp.coo_matrix(adj) if not isinstance(adj, sp.coo_matrix) else adj
        # Add self loop
        adj.setdiag(1)
        # Convert to PyTorch sparse matrix and do symmetric normalization
        self.adj_orig = adj  # Save the original copy first for dropEdge
        self.A = self._build_sparse_graph(adj)

        # 6. Constructing sparse matrix for validation/testing
        assert sp.issparse(adj_eval)
        adj_eval = sp.coo_matrix(adj_eval) if not isinstance(adj_eval, sp.coo_matrix) else adj_eval
        # Add self loop
        adj_eval.setdiag(1)
        self.adj_eval_orig = adj_eval
        self.A_eval = self._build_sparse_graph(adj_eval)


    def _build_sparse_graph(self, adj_sp):
        """
        Convert scipy.sparse.coo_matrix -> PyTorch sparse tensor and complete symmetric normalization
        A_hat = D^{-1/2} * A * D^{-1/2}
        """
        if not sp.isspmatrix_coo(adj_sp):
            adj_sp = sp.coo_matrix(adj_sp)

        # calculate degree
        row_sum = np.array(adj_sp.sum(1))  # (N,1)
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()  # (N,)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

        row, col = adj_sp.row, adj_sp.col
        data = adj_sp.data

        data = data * d_inv_sqrt[row] * d_inv_sqrt[col]

        indices = np.vstack((row, col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(data)
        shape = adj_sp.shape

        A_sp = torch.sparse_coo_tensor(i, v, torch.Size(shape)).coalesce()
        return A_sp

    def dropEdge(self):
        """
        Randomly discard some edges of self.adj_orig, then recalculate and update self.A
        """
        # Only drop the edge of the upper triangle, then sym + I
        upper = sp.triu(self.adj_orig, k=1)
        n_edge = upper.nnz
        # Number of edges retained
        n_edge_left = int((1 - self.dropedge) * n_edge)
        # Randomly choose which edges to keep
        idx_left = np.random.choice(n_edge, n_edge_left, replace=False)

        data = upper.data[idx_left]
        row = upper.row[idx_left]
        col = upper.col[idx_left]

        # Reconstruct the new adjacency matrix (COO)
        adj_new = sp.coo_matrix((data, (row, col)), shape=self.adj_orig.shape)
        # symmetry
        adj_new = adj_new + adj_new.T
        # Add self loop
        adj_new.setdiag(1)

        # Update self.A
        self.A = self._build_sparse_graph(adj_new)

    def fit(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        features = self.features.to(self.device)
        labels = self.labels.to(self.device)
        A = self.A.to(self.device)
        A_eval = self.A_eval.to(self.device)

        if len(self.labels.size()) == 2:
            nc_criterion = nn.BCEWithLogitsLoss()  # multiple tags
        else:
            nc_criterion = nn.CrossEntropyLoss()    # single label

        best_vali_acc = 0.0
        best_logits = None

        for epoch in range(self.epochs):
            # If dropedge is used, a new A is constructed
            if self.dropedge > 0:
                self.dropEdge()
                A = self.A.to(self.device)

            self.model.train()
            logits = self.model(features, A)
            loss = nc_criterion(logits[self.train_nid], labels[self.train_nid])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # verify
            self.model.eval()
            with torch.no_grad():
                logits_eval = self.model(features, A_eval).cpu()

            vali_accuracy, vali_precision, vali_recall, vali_f1 = self.eval_node_cls(
                logits_eval[self.val_nid], self.labels[self.val_nid]
            )
            # Corresponding to the original logic, use vali_f1 as vali_acc
            vali_acc = vali_f1

            if self.print_progress:
                print(f"Epoch [{epoch+1:2d}/{self.epochs}]: loss={loss.item():.4f}, "
                      f"vali_acc={vali_acc:.4f}, vali_precision={vali_precision:.4f}, "
                      f"vali_recall={vali_recall:.4f}, vali_f1={vali_f1:.4f}")

            # If the validation set performs better, update
            if vali_acc > best_vali_acc:
                best_vali_acc = vali_acc
                best_logits = logits_eval

                # Record the test set at the same time
                test_accuracy, test_precision, test_recall, test_f1 = self.eval_node_cls(
                    logits_eval[self.test_nid], self.labels[self.test_nid]
                )
                best_test_acc = test_accuracy
                best_test_prec = test_precision
                best_test_rec = test_recall
                best_test_f1 = test_f1
                best_epoch = epoch
                if self.print_progress:
                    print(" " * 20 +
                          f"test_acc={test_accuracy:.4f}, test_prec={test_precision:.4f}, "
                          f"test_rec={test_recall:.4f}, test_f1={test_f1:.4f}")

        if self.print_progress:
            print(f"Final test result = "
                  f"test_acc={best_test_acc:.4f}, test_prec={best_test_prec:.4f}, "
                  f"test_rec={best_test_rec:.4f}, test_f1={best_test_f1:.4f}")

        # clean up
        del self.model, features, labels, A, A_eval
        torch.cuda.empty_cache()
        gc.collect()

        t_cost = time.time() - self.t
        if self.print_progress:
            print("Total time: {:.2f}s".format(t_cost))

        return best_test_acc, best_test_prec, best_test_rec, best_test_f1, best_epoch

    def eval_node_cls(self, logits, labels):
        logits = logits.cpu()
        labels = labels.cpu()

        # Single label classification
        preds = torch.argmax(logits, dim=1)  # Get the predicted label index for each node
        
        # Use 'macro' averaging to calculate precision, recall, and F1 score
        acc = accuracy_score(labels, preds)  # accuracy_score for single-label tasks
        prec = precision_score(labels, preds, average='macro', zero_division=0)
        rec = recall_score(labels, preds, average='macro', zero_division=0)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)

        return acc, prec, rec, f1
    # =======================
    # 2. GCN layers used in the model  
    # =======================
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None, dropout=0.0, bias=True):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats

        # parameter
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None

        # activation function
        self.activation = activation
        # dropout
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        self.reset_parameters()

    def reset_parameters(self):
        # Simple uniform initialization
        stdv = 1.0 / math.sqrt(self.out_feats)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, X, A):
        """
        X: [N, in_feats]
        A: 稀疏矩阵 (N,N) ，已经是归一化过的 A_hat
        """
        if self.dropout:
            X = self.dropout(X)

        # (N, in_feats) x (in_feats, out_feats) -> (N, out_feats)
        XW = X @ self.weight

        # Sparse multiplication: (N, N) * (N, out_feats) -> (N, out_feats)
        out = torch.sparse.mm(A, XW)

        if self.bias is not None:
            out = out + self.bias

        if self.activation:
            out = self.activation(out)
        return out


# =======================
# 3. GCN model stacking
# =======================
class GCN_model(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()

        self.layers = nn.ModuleList()
        # Layer 1 (input layer) does not perform dropout
        self.layers.append(GCNLayer(in_feats, n_hidden, activation=activation, dropout=0.0))

        # middle hidden layer
        for _ in range(n_layers - 1):
            self.layers.append(GCNLayer(n_hidden, n_hidden, activation=activation, dropout=dropout))

        # Output layer: The activation function can be omitted
        self.layers.append(GCNLayer(n_hidden, n_classes, activation=None, dropout=dropout))

    def forward(self, X, A):
        for layer in self.layers:
            X = layer(X, A)
        return X

import gc
import sys
import math
import time
import pickle
import networkx as nx
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

# ★ sklearn.metrics Used to calculate acc, prec, rec, f1
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


###############################################################################
# ☆ Custom GraphSAGE convolutional layer to replace dgl.nn.pytorch.conv.SAGEConv
#   We only implement aggregation method with aggregator_type='gcn'
###############################################################################
class GraphSAGELayer(nn.Module):
    def __init__(self, in_feats, out_feats, aggregator_type='gcn',
                 feat_drop=0.0, activation=None):
        super(GraphSAGELayer, self).__init__()
        # Linear transformation to map the aggregated features to the out_feats dimension
        self.fc = nn.Linear(in_feats, out_feats, bias=True)
        # Feature dropout layer
        self.feat_drop = nn.Dropout(feat_drop)
        # activation function
        self.activation = activation
        # Aggregator type, only 'gcn' is implemented here
        self.aggregator_type = aggregator_type

    def forward(self, adj_sp, h):
        """
        adj_sp: the normalized graph adjacency matrix passed in (torch.sparse_coo_tensor or other sparse formats)
        h     : node features, size (N, in_feats)
        """
        # First, perform dropout on the features
        h = self.feat_drop(h)
        # According to aggregator_type='gcn', we execute: h_agg = A_norm * h
        # where A_norm is the symmetric normalized adjacency matrix
        h_agg = torch.sparse.mm(adj_sp, h)

        # linear transformation
        h_agg = self.fc(h_agg)

        # If an activation function is set, execute
        if self.activation is not None:
            h_agg = self.activation(h_agg)

        return h_agg


###############################################################################
# ☆ The main body of the Jumping Knowledge Network (JKNet) model
#   Several layers of GraphSAGEConv (here replaced by GraphSAGELayer) are stacked
#   And at the output, the results of each layer are concatenated (concat) and finally connected to a linear layer
###############################################################################
class JKNet_model(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type='gcn'):
        super(JKNet_model, self).__init__()
        # Used to store multi-layer GraphSAGE convolutions
        self.layers = nn.ModuleList()

        # First layer: feat_drop is set to 0 based on the original code.
        self.layers.append(
            GraphSAGELayer(in_feats, n_hidden, aggregator_type=aggregator_type,
                           feat_drop=0.0, activation=activation)
        )

        # Middle layers: feat_drop is set to dropout
        for i in range(n_layers - 1):
            self.layers.append(
                GraphSAGELayer(n_hidden, n_hidden, aggregator_type=aggregator_type,
                               feat_drop=dropout, activation=activation)
            )

        # After the final concatenation (n_layers output concatenation), it is mapped to n_classes dimension
        self.layer_output = nn.Linear(n_hidden * n_layers, n_classes)

    def forward(self, adj_sp, features):
        # h is used to iteratively store the features of the current layer
        h = features
        # hs is used to collect the features of each layer for final splicing
        hs = []

        # GraphSAGE aggregation layer by layer
        for layer in self.layers:
            h = layer(adj_sp, h)
            hs.append(h)

        # Concatenate the output features of all layers on the feature dimension
        h_cat = torch.cat(hs, dim=1)

        # The classification result is obtained through the final linear layer
        out = self.layer_output(h_cat)
        return out


###############################################################################
# ☆ JKNet main class, managing data, model construction and training process
###############################################################################
class JKNet():
    def __init__(self,
                 adj,          # Adjacency matrix for training (scipy sparse matrix)
                 adj_eval,     # Adjacency matrix for inference/evaluation (scipy sparse matrix)
                 features,     # Node characteristics
                 labels,       # Node label
                 tvt_nids,     # Training/validation/testing node index
                 cuda=-1,
                 hidden_size=128,
                 n_layers=3,
                 epochs=200,
                 seed=-1,
                 lr=1e-2,
                 weight_decay=5e-4,
                 dropout=0.5,
                 print_progress=True,
                 dropedge=0):
        # Initialization start time, used to observe program time consumption
        self.t = time.time()
        # learning rate
        self.lr = lr
        # Weight attenuation coefficient
        self.weight_decay = weight_decay
        # Number of training rounds
        self.epochs = epochs
        # Whether to print training progress
        self.print_progress = print_progress
        # dropedge ratio
        self.dropedge = dropedge

        # Configure the device to use the CPU if no GPU is available
        if not torch.cuda.is_available():
            cuda = -1
        self.device = torch.device(f'cuda:{cuda % 8}' if cuda >= 0 else 'cpu')

        # Fixed random seed to ensure reproducibility (if seed>0)
        if seed > 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Load data and do corresponding initialization processing
        self.load_data(adj, adj_eval, features, labels, tvt_nids)

        # Build model
        self.model = JKNet_model(
            in_feats=self.features.size(1), # Input dimension = feature dimension
            n_hidden=hidden_size,           # Hidden layer size
            n_classes=self.n_class,         # Number of categories
            n_layers=n_layers,             # Number of layers
            activation=F.relu,             # activation function
            dropout=dropout,               # dropout
            aggregator_type='gcn'          # Default gcn aggregation mode
        )
        # Put the model on the specified device
        self.model.to(self.device)


    ############################################################################
    # ☆ Convert scipy's sparse matrix to pytorch's sparse matrix and perform symmetric normalization
    ############################################################################
    def normalize_adj_torch(self, adj_sp):
        """
        Input: adj_sp (scipy.sparse), shape (N, N), assumed to contain self-loops (diagonal=1)
        Output: adj_norm_torch (torch.sparse), shape (N, N)
        """
        # Convert adj_sp to CSR format for easy row access
        adj_csr = sp.csr_matrix(adj_sp)
        # Computational degree
        deg = np.array(adj_csr.sum(1))  # shape = (N,1)
        # Calculate D^-1/2
        d_inv_sqrt = np.power(deg, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

        # D^-1/2 is put into the diagonal matrix
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        # Calculate the symmetric normalization A_norm = D^-1/2 * A * D^-1/2
        adj_norm = d_inv_sqrt_mat.dot(adj_csr).dot(d_inv_sqrt_mat)

        # Convert to COO format for easy construction of torch.sparse
        adj_coo = sp.coo_matrix(adj_norm)
        # Constructing indices
        indices = torch.LongTensor([adj_coo.row, adj_coo.col])
        # Construct values
        values = torch.FloatTensor(adj_coo.data)
        # shape
        shape = adj_coo.shape

        # Create sparse tensors with torch.sparse_coo_tensor
        adj_norm_torch = torch.sparse_coo_tensor(indices, values, torch.Size(shape))
        return adj_norm_torch


    ############################################################################
    # ☆ Load the data and prepare the adjacency matrix for training and evaluation
    ############################################################################
    def load_data(self, adj, adj_eval, features, labels, tvt_nids):
        # Convert features to FloatTensor
        if isinstance(features, torch.FloatTensor) or isinstance(features, torch.cuda.FloatTensor):
            self.features = features
        else:
            self.features = torch.FloatTensor(features)

        # No longer perform L1 normalization on features

        # Convert labels to Tensor
        if len(labels.shape) == 2:
            labels = torch.FloatTensor(labels)
        else:
            labels = torch.LongTensor(labels)
        self.labels = labels

        # If the label is one-dimensional (multi-classification), then the number of categories = the number of unique(labels)
        # If the label is two-dimensional (multi-label), the number of categories = label.size(1)
        if len(self.labels.size()) == 1:
            self.n_class = len(torch.unique(self.labels))
        else:
            self.n_class = self.labels.size(1)

        # Get the node numbers for training, validation, and testing (from the dictionary)
        self.train_nid = tvt_nids['train']
        self.val_nid = tvt_nids['val']
        self.test_nid = tvt_nids['test']

        # Adjacency matrix for training (with self-loops)
        assert sp.issparse(adj)
        if not isinstance(adj, sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        adj.setdiag(1)
        self.adj = adj  # scipy coo_matrix

        # Adjacency matrix for inference/evaluation (with self-loops)
        assert sp.issparse(adj_eval)
        if not isinstance(adj_eval, sp.coo_matrix):
            adj_eval = sp.coo_matrix(adj_eval)
        adj_eval.setdiag(1)
        self.adj_eval = adj_eval  # scipy coo_matrix

        # Normalize the adjacency matrix for pre-evaluation, save as self.adj_eval_norm
        self.adj_eval_norm = self.normalize_adj_torch(self.adj_eval).to(self.device)
        # If you don't dropedge during training, just prepare self.adj_norm directly
        if self.dropedge <= 0:
            self.adj_norm = self.normalize_adj_torch(self.adj).to(self.device)


    ############################################################################
    # ☆ Randomly discard some edges to get the adjacency matrix after dropedge and save it as self.adj_norm
    ############################################################################
    def dropEdge(self):
        # First take the upper triangle
        upper = sp.triu(self.adj, 1)
        # Original number of sides (excluding diagonals)
        n_edge = upper.nnz
        # The number of edges retained
        n_edge_left = int((1 - self.dropedge) * n_edge)
        # Randomly select n_edge_left edges to keep
        index_edge_left = np.random.choice(n_edge, n_edge_left, replace=False)
        # Get the retained edges
        data = upper.data[index_edge_left]
        row = upper.row[index_edge_left]
        col = upper.col[index_edge_left]
        # Reconstruct the upper triangle of the adjacency matrix
        adj = sp.coo_matrix((data, (row, col)), shape=self.adj.shape)
        # Symmetrization
        adj = adj + adj.T
        # Add self-loop
        adj.setdiag(1)

        # Call normalize_adj_torch for normalization
        self.adj_norm = self.normalize_adj_torch(adj).to(self.device)


    ############################################################################
    # ☆ Model training function
    #   Need to return: best_test_acc, best_test_prec, best_test_rec, best_test_f1, best_epoch
    ############################################################################
    def fit(self):
        # Defining the optimizer (Adam)
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        # Put features and labels on device
        features = self.features.to(self.device)
        labels = self.labels.to(self.device)

        # Use CrossEntropyLoss as the loss function (single-label task)
        nc_criterion = nn.CrossEntropyLoss()

        # For recording the best verification micro-f1
        best_vali_f1 = 0.0
        # Used to save the four indicators of the corresponding test set
        best_test_acc = 0.0
        best_test_prec = 0.0
        best_test_rec = 0.0
        best_test_f1 = 0.0
        # Record the best epoch
        best_epoch = 0

        # train for several rounds
        for epoch in range(self.epochs):
            # If dropedge is needed, the edge must be randomly dropped again in each epoch
            if self.dropedge > 0:
                self.dropEdge()
            # Otherwise use the pre-stored adj_norm
            else:
                pass

            # training mode
            self.model.train()
            # Forward propagation (training adjacency using self.adj_norm)
            logits = self.model(self.adj_norm, features)
            # Calculate the loss on the training set (only for train_nid)
            loss_train = nc_criterion(logits[self.train_nid], labels[self.train_nid])

            # Backpropagation
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            # Verification: No need for dropout, etc., set model.eval()
            self.model.eval()
            with torch.no_grad():
                # Adjacency for evaluation (self.adj_eval_norm), forward pass
                logits_eval = self.model(self.adj_eval_norm, features)
            # Calculate the metrics for the validation set
            vali_acc, vali_prec, vali_rec, vali_f1 = self.eval_node_cls(
                logits_eval[self.val_nid], labels[self.val_nid]
            )

            if self.print_progress:
                print(f"Epoch [{epoch+1:2d}/{self.epochs}]: loss: {loss_train.item():.4f}, "
                      f"vali_acc: {vali_acc:.4f}, vali_prec: {vali_prec:.4f}, vali_rec: {vali_rec:.4f}, vali_f1: {vali_f1:.4f}")

            # If the validation set f1 performs better, evaluate and update the best metric on the test set
            if vali_f1 > best_vali_f1:
                best_vali_f1 = vali_f1
                best_epoch = epoch
                # Calculate four indicators on the test set
                test_acc, test_prec, test_rec, test_f1 = self.eval_node_cls(
                    logits_eval[self.test_nid], labels[self.test_nid]
                )
                best_test_acc = test_acc
                best_test_prec = test_prec
                best_test_rec = test_rec
                best_test_f1 = test_f1
                if self.print_progress:
                    print(f"                 test_acc: {test_acc:.4f}, "
                          f"test_prec: {test_prec:.4f}, test_rec: {test_rec:.4f}, "
                          f"test_f1: {test_f1:.4f}")

        # After the training is completed, print the final results
        if self.print_progress:
            print("Final test results:")
            print(f"acc: {best_test_acc:.4f}, prec: {best_test_prec:.4f}, "
                  f"rec: {best_test_rec:.4f}, f1: {best_test_f1:.4f}")

        # Free up GPU memory
        del self.model, features, labels, self.adj_norm, self.adj_eval_norm
        torch.cuda.empty_cache()
        gc.collect()

        # Returns the best test set metric and the best epoch
        return best_test_acc, best_test_prec, best_test_rec, best_test_f1, best_epoch


    ############################################################################
    # ☆ Evaluation node classification indicators: acc, prec, rec, f1
    ############################################################################
    def eval_node_cls(self, logits, labels):
        """
        logits: Dimension (N, n_class), raw value of the prediction
        labels: Dimension (N,), true label
        """
        # Single label classification: take the maximum index
        preds = torch.argmax(logits, dim=1)

        # Go to CPU, convert to numpy for sklearn metrics
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()

        # Calculate four indicators
        acc = accuracy_score(labels_np, preds_np)
        prec = precision_score(labels_np, preds_np, average='macro', zero_division=0)
        rec = recall_score(labels_np, preds_np, average='macro', zero_division=0)
        f1 = f1_score(labels_np, preds_np, average='macro', zero_division=0)

        return acc, prec, rec, f1
import gc  # Import the gc module to manually release memory
import sys  # Import the sys module, generally used for system-level operations
import math  # Mathematical calculation module
import time  # Time-related modules for calculating training or inference time
import pickle  # Binary serialization for storing and loading data
import networkx as nx  # Libraries for processing graph structures (optional here, used to view or process graphs)
import numpy as np  # Numerical calculation library
import scipy.sparse as sp  # Sparse matrix manipulation library
import torch  # PyTorch main library
import torch.nn as nn  # Neural network related modules
import torch.nn.functional as F  # Common activation functions, loss and other functions
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score  # Introducing common evaluation indicators

##############################################################################
# The following function is used to convert the scipy sparse adjacency matrix into a sparse tensor in PyTorch and perform standard GCN normalization (D^-1/2 * A * D^-1/2).
##############################################################################
def normalize_adj(adj):
    # adj is a scipy.sparse sparse matrix (either coo or csr)

    if not isinstance(adj, sp.coo_matrix):
        # If adj is not in coo format, convert it to coo format
        adj = adj.tocoo()

    # Get the row index, column index and corresponding data from the coo matrix
    row = torch.from_numpy(adj.row).long()      # Starting vertex of adjacent edge
    col = torch.from_numpy(adj.col).long()      # End vertex of adjacent edge
    data = torch.from_numpy(adj.data).float()   # The weight of each edge (usually 1)

    # The total number of nodes in the graph
    n = adj.shape[0]

    # Calculate the degree of each node(deg). In the coo format, row stores the starting point of the edge.
    # We can get the degree of the node by aggregating the values ​​in data
    deg = torch.zeros(n, dtype=torch.float)
    deg.index_add_(0, row, data)  # Add the degree of the node corresponding to row[i] to data[i]

    # Calculates deg raised to the power of -1/2
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # Replace infinity with 0 to prevent division by 0

    # Scale data by D^-1/2 * data * D^-1/2
    # That is, the weight of edge (u, v) = deg_inv_sqrt[u] * w(u,v) * deg_inv_sqrt[v]
    data = deg_inv_sqrt[row] * data * deg_inv_sqrt[col]

    # Construct a PyTorch sparse matrix (n x n) with new data, row, col
    A = torch.sparse_coo_tensor(
        indices=torch.stack([row, col], dim=0),
        values=data,
        size=(n, n)
    )
    # coalesce() will merge duplicate indices and keep indices in order to ensure the validity of sparse matrix representation
    A = A.coalesce()
    return A

##############################################################################
# Define a GCNLayer class to simulate the SAGEConv layer logic of aggregator_type='gcn' in DGL.
# The core operation here is A * X * W, where A is the normalized adjacency matrix (A_hat),
# X is the input features and W is the learnable weights.
##############################################################################
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None, dropout=0.0):
        super(GCNLayer, self).__init__()
        # Define the learnable parameters W and b
        self.weight = nn.Parameter(torch.FloatTensor(in_feats, out_feats))
        self.bias = nn.Parameter(torch.FloatTensor(out_feats))

        # The activation function can be relu, etc. If it is None, no activation is used
        self.activation = activation
        # Dropout layer, used for random inactivation of features to prevent overfitting
        self.dropout = nn.Dropout(dropout)

        # Initialize the parameters
        nn.init.xavier_uniform_(self.weight)  # Xavier initialization makes weight distribution more stable
        nn.init.zeros_(self.bias)             # The bias is initialized to 0

    def forward(self, A, X):
        # X first goes through dropout
        X = self.dropout(X)
        # Sparse multiplication: A is a (N x N) sparse matrix, X is (N x in_feats)
        # Compute A * X = (N x in_feats)
        h = torch.sparse.mm(A, X)
        # Linear transformation: h * W + b = (N x out_feats)
        h = h @ self.weight + self.bias
        # If there is an activation function, further processing
        if self.activation is not None:
            h = self.activation(h)
        return h

##############################################################################
# GraphSAGE_model corresponds to the GraphSAGE_model class in the original code, but no longer uses dgl's SAGEConv,
# Instead, use the custom GCNLayer above to implement the "gcn" aggregation method.
##############################################################################
class GraphSAGE_model(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type='gcn'):
        super(GraphSAGE_model, self).__init__()
        # Here we use ModuleList to stack multiple layers of GCNLayer
        self.layers = nn.ModuleList()

        # The first layer (input layer), in_feats -> n_hidden
        # The dropout here is set to 0 to simulate the original first layer SAGEConv without feat_drop
        self.layers.append(
            GCNLayer(
                in_feats=in_feats,
                out_feats=n_hidden,
                activation=activation,
                dropout=0.0
            )
        )

        # If n_layers=1, the loop will not be entered.
        # If n_layers>1, add an intermediate hidden layer (n_layers-1)
        for i in range(n_layers - 1):
            self.layers.append(
                GCNLayer(
                    in_feats=n_hidden,
                    out_feats=n_hidden,
                    activation=activation,
                    dropout=dropout
                )
            )

        # Output layer, n_hidden -> n_classes, without activation function
        self.layers.append(
            GCNLayer(
                in_feats=n_hidden,
                out_feats=n_classes,
                activation=None,
                dropout=dropout
            )
        )

    def forward(self, A, features):
        # Feature initial input
        h = features
        # Pass through each layer of GCNLayer in turn
        for layer in self.layers:
            h = layer(A, h)
        # return final output
        return h

##############################################################################
# GraphSAGE main class, responsible for data loading, model training, evaluation and other functions
##############################################################################
class GraphSAGE(object):
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
        # Record the start time to count the training time
        self.t = time.time()

        # learning rate
        self.lr = lr
        # Weight attenuation coefficient
        self.weight_decay = weight_decay
        # Number of training iterations
        self.epochs = epochs
        # Whether to print the training process
        self.print_progress = print_progress
        # Whether to perform dropEdge and the ratio of dropEdge
        self.dropedge = dropedge

        # Device selection: If GPU is available and cuda>=0, use cuda, otherwise use cpu
        if not torch.cuda.is_available():
            cuda = -1
        self.device = torch.device(f'cuda:{cuda%8}' if cuda >= 0 else 'cpu')

        # Random seed fixed (if seed>0)
        if seed > 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Loading/Processing Data
        self.load_data(adj, adj_eval, features, labels, tvt_nids)

        # Initialize model
        self.model = GraphSAGE_model(
            in_feats=self.features.size(1),
            n_hidden=hidden_size,
            n_classes=self.n_class,
            n_layers=n_layers,
            activation=F.relu,
            dropout=dropout,
            aggregator_type='gcn'  # This can be set arbitrarily, but we currently only implement the gcn method
        )
        # Place the model on the specified device
        self.model.to(self.device)

    def load_data(self, adj, adj_eval, features, labels, tvt_nids):
        # Convert features to FloatTensor
        if isinstance(features, torch.FloatTensor) or isinstance(features, torch.cuda.FloatTensor):
            self.features = features
        else:
            self.features = torch.FloatTensor(features)

        # No longer perform L1 normalization on feature dimensions (1433, 3703)

        # Convert the label to the corresponding tensor
        if len(labels.shape) == 2:
            labels = torch.FloatTensor(labels)
        else:
            labels = torch.LongTensor(labels)
        self.labels = labels

        # Determine whether it is multi-label or single-label, and determine the number of categories
        if len(self.labels.size()) == 1:
            self.n_class = len(torch.unique(self.labels))
        else:
            self.n_class = labels.size(1)

        # Node ids for splitting training/validation/testing
        self.train_nid = tvt_nids['train']
        self.val_nid = tvt_nids['val']
        self.test_nid = tvt_nids['test']

        # -------------------------------------
        # Processing the adjacency matrix for training
        # -------------------------------------
        assert sp.issparse(adj), "adj Must be in sparse matrix format"
        if not isinstance(adj, sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        # Add self loop
        adj.setdiag(1)
        # Convert to csr for easy subsequent processing
        adj = sp.csr_matrix(adj)
        # Save the original sparse matrix (dropEdge may be needed during training, so keep the original adj)
        self.adj = adj
        # Normalize once and save as a PyTorch sparse tensor
        self.A = normalize_adj(adj)
        # Put it on the designated device
        self.A = self.A.to(self.device)

        # -------------------------------------
        # Processing adjacency matrices for inference/validation
        # -------------------------------------
        assert sp.issparse(adj_eval), "adj_eval must be in sparse matrix format"
        if not isinstance(adj_eval, sp.coo_matrix):
            adj_eval = sp.coo_matrix(adj_eval)
        # Add self loop
        adj_eval.setdiag(1)
        # Convert to csr
        adj_eval = sp.csr_matrix(adj_eval)
        self.adj_eval = adj_eval
        # Save after normalization
        self.A_eval = normalize_adj(adj_eval)
        self.A_eval = self.A_eval.to(self.device)

    def dropEdge(self):
        # Randomly remove some edges in the original adj (upper triangle)
        # The purpose is to prevent overfitting, similar to dropout but acting on the edges of the graph

        # Extract the upper triangle
        upper = sp.triu(self.adj, 1)
        # Total number of edges (excluding diagonals and lower triangle repeating parts)
        n_edge = upper.nnz
        # Number of remaining sides
        n_edge_left = int((1 - self.dropedge) * n_edge)

        # Randomly select edges to keep (index)
        index_edge_left = np.random.choice(n_edge, n_edge_left, replace=False)

        # According to these indexes, the corresponding rows, columns and weights are retrieved.
        data = upper.data[index_edge_left]
        row = upper.row[index_edge_left]
        col = upper.col[index_edge_left]

        # Construct a new upper triangular sparse matrix
        adj = sp.coo_matrix((data, (row, col)), shape=self.adj.shape)
        # Symmetrically, we get the complete adjacency matrix
        adj = adj + adj.T
        # Add self loop
        adj.setdiag(1)

        # Standardize the new adj and save it to self.A (dedicated to training)
        self.A = normalize_adj(adj)
        self.A = self.A.to(self.device)

    def fit(self):
        # Define the optimizer, use Adam, and have L2 regularization (weight_decay)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # Moving features and labels to computing devices
        features = self.features.to(self.device)
        labels = self.labels.to(self.device)

        # Single label classification, using CrossEntropyLoss
        nc_criterion = nn.CrossEntropyLoss()

        # Used to record the best validation set accuracy
        best_vali_acc = 0.0

        # Store the four best test sets
        best_test_acc = 0.0
        best_test_prec = 0.0
        best_test_rec = 0.0
        best_test_f1 = 0.0

        # Record the best epoch
        best_epoch = 0

        # best_logits is used for visualization or analysis, but here it is mainly used to save when the validation set is optimal
        best_logits = None

        # Start iterative training
        for epoch in range(self.epochs):
            # If dropedge>0 is set, drop again
            if self.dropedge > 0:
                self.dropEdge()

            # Enter training mode
            self.model.train()

            # Forward propagation: Calculate the logits output by the current model
            logits = self.model(self.A, features)

            # Calculate the current training loss (only on the training set node)
            loss_train = nc_criterion(logits[self.train_nid], labels[self.train_nid])

            # Back propagation and updating parameters
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            # Switch to evaluation mode without dropout enabled
            self.model.eval()
            with torch.no_grad():
                # Use the complete evaluation adjacency matrix A_eval to get the logits
                logits_eval = self.model(self.A_eval, features).detach().cpu()

            # Calculate the evaluation index on the validation set (mainly look at acc to decide whether to update the optimal)
            vali_acc, vali_prec, vali_rec, vali_f1 = self.eval_node_cls(
                logits_eval[self.val_nid],
                labels[self.val_nid].cpu()
            )

            # If you need to print training process information
            if self.print_progress:
                print(f"Epoch [{epoch+1:2d}/{self.epochs}]: loss: {loss_train.item():.4f}, "
                      f"vali acc: {vali_acc:.4f}, vali_prec: {vali_prec:.4f}, vali_rec: {vali_rec:.4f}, vali_f1: {vali_f1:.4f}")

            # If the acc on the validation set is higher, update the best record and evaluate it on the test set.
            if vali_acc > best_vali_acc:
                best_vali_acc = vali_acc
                best_logits = logits_eval
                best_epoch = epoch

                # Calculate four indicators on the test set
                test_acc, test_prec, test_rec, test_f1 = self.eval_node_cls(
                    best_logits[self.test_nid],
                    labels[self.test_nid].cpu()
                )

                best_test_acc = test_acc
                best_test_prec = test_prec
                best_test_rec = test_rec
                best_test_f1 = test_f1

                if self.print_progress:
                    print(f"                 >>> test acc: {test_acc:.4f}, "
                          f"prec: {test_prec:.4f}, rec: {test_rec:.4f}, f1: {test_f1:.4f}")

        # Print the final test results (based on the best validation set)
        if self.print_progress:
            print("Final test results: "
                  f"acc: {best_test_acc:.4f}, prec: {best_test_prec:.4f}, "
                  f"rec: {best_test_rec:.4f}, f1: {best_test_f1:.4f}")

        # free cache
        del self.model, features, labels, self.A
        torch.cuda.empty_cache()
        gc.collect()

        # Calculate the total training time
        t_used = time.time() - self.t

        # Returns four metrics: acc, prec, rec, f1 and best epoch
        return best_test_acc, best_test_prec, best_test_rec, best_test_f1, best_epoch

    def eval_node_cls(self, logits, labels):
        # Single-label classification: direct argmax
        preds = torch.argmax(logits, dim=1)
        average_type = 'macro'

        # Convert to numpy for easy use with sklearn
        preds_np = preds.numpy()
        labels_np = labels.numpy()

        # Calculate four indicators
        acc = accuracy_score(labels_np, preds_np)
        prec = precision_score(labels_np, preds_np, average=average_type, zero_division=0)
        rec = recall_score(labels_np, preds_np, average=average_type, zero_division=0)
        f1 = f1_score(labels_np, preds_np, average=average_type, zero_division=0)

        # Return four indicators
        return acc, prec, rec, f1
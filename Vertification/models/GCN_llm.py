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
# 1. 模型的整体类 (不使用DGL)
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
        参数基本同原来，只是去掉了对DGLGraph的依赖，并且去掉对 1433/3703 维度特征做 L1 归一化的逻辑
        """
        self.t = time.time()               # 记录开始时间
        self.lr = lr                       # 学习率
        self.weight_decay = weight_decay   # L2正则化系数
        self.epochs = epochs               # 训练轮数
        self.print_progress = print_progress
        self.dropedge = dropedge           # 边丢弃率

        # 配置计算设备 (CPU/GPU)
        if not torch.cuda.is_available():
            cuda = -1
        self.device = torch.device(f'cuda:{cuda%8}' if cuda >= 0 else 'cpu')

        # 设置随机种子
        if seed > 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # 数据加载和预处理
        self.load_data(adj, adj_eval, features, labels, tvt_nids)

        # 初始化模型
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
        不使用DGL，手动处理邻接矩阵，存成稀疏的 PyTorch 矩阵（用于训练和评估）
        """

        # 1. 处理特征 (已经是从 LLM 得到的向量，不再做归一化)
        if isinstance(features, torch.FloatTensor) or isinstance(features, torch.cuda.FloatTensor):
            self.features = features
        else:
            self.features = torch.FloatTensor(features)

        # 2. 处理标签
        if len(labels.shape) == 2:
            labels = torch.FloatTensor(labels)  # 多标签
        else:
            labels = torch.LongTensor(labels)   # 单标签
        self.labels = labels

        # 3. 确定类别数
        if len(self.labels.size()) == 1:
            self.n_class = len(torch.unique(self.labels))  # 单标签：种类数
        else:
            self.n_class = labels.size(1)                  # 多标签：向量长度

        # 4. 划分训练/验证/测试集
        self.train_nid = tvt_nids['train']
        self.val_nid = tvt_nids['val']
        self.test_nid = tvt_nids['test']

        # 5. 构建训练用的稀疏矩阵
        assert sp.issparse(adj)
        adj = sp.coo_matrix(adj) if not isinstance(adj, sp.coo_matrix) else adj
        # 添加自环
        adj.setdiag(1)
        # 转化为 PyTorch 稀疏矩阵并做对称归一化
        self.adj_orig = adj  # 先存一份原始的，用于dropEdge
        self.A = self._build_sparse_graph(adj)

        # 6. 构建验证/测试用的稀疏矩阵
        assert sp.issparse(adj_eval)
        adj_eval = sp.coo_matrix(adj_eval) if not isinstance(adj_eval, sp.coo_matrix) else adj_eval
        # 添加自环
        adj_eval.setdiag(1)
        self.adj_eval_orig = adj_eval
        self.A_eval = self._build_sparse_graph(adj_eval)


    def _build_sparse_graph(self, adj_sp):
        """
        将 scipy.sparse.coo_matrix -> PyTorch 稀疏张量，并完成对称归一化
        A_hat = D^{-1/2} * A * D^{-1/2}
        """
        if not sp.isspmatrix_coo(adj_sp):
            adj_sp = sp.coo_matrix(adj_sp)

        # 计算度
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
        对 self.adj_orig 进行随机丢弃部分边，再重新计算并更新 self.A
        """
        # 只对上三角丢边，然后再 sym + I
        upper = sp.triu(self.adj_orig, k=1)
        n_edge = upper.nnz
        # 保留的边数
        n_edge_left = int((1 - self.dropedge) * n_edge)
        # 随机选择保留哪些边
        idx_left = np.random.choice(n_edge, n_edge_left, replace=False)

        data = upper.data[idx_left]
        row = upper.row[idx_left]
        col = upper.col[idx_left]

        # 重构新的邻接矩阵 (COO)
        adj_new = sp.coo_matrix((data, (row, col)), shape=self.adj_orig.shape)
        # 对称
        adj_new = adj_new + adj_new.T
        # 添加自环
        adj_new.setdiag(1)

        # 更新 self.A
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
            nc_criterion = nn.BCEWithLogitsLoss()  # 多标签
        else:
            nc_criterion = nn.CrossEntropyLoss()    # 单标签

        best_vali_acc = 0.0
        best_logits = None

        for epoch in range(self.epochs):
            # 若使用dropedge，则构建新的 A
            if self.dropedge > 0:
                self.dropEdge()
                A = self.A.to(self.device)

            self.model.train()
            logits = self.model(features, A)
            loss = nc_criterion(logits[self.train_nid], labels[self.train_nid])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 验证
            self.model.eval()
            with torch.no_grad():
                logits_eval = self.model(features, A_eval).cpu()

            vali_accuracy, vali_precision, vali_recall, vali_f1 = self.eval_node_cls(
                logits_eval[self.val_nid], self.labels[self.val_nid]
            )
            # 与原逻辑对应，用 vali_f1 作为 vali_acc
            vali_acc = vali_f1

            if self.print_progress:
                print(f"Epoch [{epoch+1:2d}/{self.epochs}]: loss={loss.item():.4f}, "
                      f"vali_acc={vali_acc:.4f}, vali_precision={vali_precision:.4f}, "
                      f"vali_recall={vali_recall:.4f}, vali_f1={vali_f1:.4f}")

            # 如果验证集表现更好，则更新
            if vali_acc > best_vali_acc:
                best_vali_acc = vali_acc
                best_logits = logits_eval

                # 同时记录测试集
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

        # 清理
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

        # 单标签分类
        preds = torch.argmax(logits, dim=1)  # 获取每个节点的预测标签索引
        
        # 使用 'macro' 平均来计算精确度、召回率、F1分数
        acc = accuracy_score(labels, preds)  # accuracy_score 用于单标签任务
        prec = precision_score(labels, preds, average='macro', zero_division=0)
        rec = recall_score(labels, preds, average='macro', zero_division=0)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)

        return acc, prec, rec, f1
    # =======================
# 2. 模型中用到的 GCN 层
# =======================
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None, dropout=0.0, bias=True):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats

        # 参数
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None

        # 激活函数
        self.activation = activation
        # dropout
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        self.reset_parameters()

    def reset_parameters(self):
        # 简单的均匀初始化
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

        # 稀疏相乘: (N, N) * (N, out_feats) -> (N, out_feats)
        out = torch.sparse.mm(A, XW)

        if self.bias is not None:
            out = out + self.bias

        if self.activation:
            out = self.activation(out)
        return out


# =======================
# 3. GCN 模型堆叠
# =======================
class GCN_model(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()

        self.layers = nn.ModuleList()
        # 第1层 (输入层) 不做dropout
        self.layers.append(GCNLayer(in_feats, n_hidden, activation=activation, dropout=0.0))

        # 中间隐藏层
        for _ in range(n_layers - 1):
            self.layers.append(GCNLayer(n_hidden, n_hidden, activation=activation, dropout=dropout))

        # 输出层: 激活函数可以省略
        self.layers.append(GCNLayer(n_hidden, n_classes, activation=None, dropout=dropout))

    def forward(self, X, A):
        for layer in self.layers:
            X = layer(X, A)
        return X

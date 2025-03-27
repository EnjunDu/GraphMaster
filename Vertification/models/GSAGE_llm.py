import gc  # 导入gc模块，用于手动释放内存
import sys  # 导入sys模块，一般用于系统级别操作
import math  # 数学计算模块
import time  # 时间相关模块，用于计算训练或推断时间
import pickle  # 用于存储和加载数据的二进制序列化
import networkx as nx  # 处理图结构的库(此处可选，用于查看或处理图)
import numpy as np  # 数值计算库
import scipy.sparse as sp  # 稀疏矩阵操作库
import torch  # PyTorch 主库
import torch.nn as nn  # 神经网络相关模块
import torch.nn.functional as F  # 常用激活函数、loss等函数
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score  # 引入常用评价指标

##############################################################################
# 下面这个函数用于将scipy的稀疏邻接矩阵转为PyTorch中的稀疏张量，并进行标准的GCN规范化(D^-1/2 * A * D^-1/2)。
##############################################################################
def normalize_adj(adj):
    # adj 是一个 scipy.sparse 的稀疏矩阵（格式coo或csr均可）

    if not isinstance(adj, sp.coo_matrix):
        # 如果adj不是coo格式，就转为coo格式
        adj = adj.tocoo()

    # 从coo矩阵中获取行索引、列索引和对应的数据
    row = torch.from_numpy(adj.row).long()      # 邻接边的起始顶点
    col = torch.from_numpy(adj.col).long()      # 邻接边的结束顶点
    data = torch.from_numpy(adj.data).float()   # 每条边上的权重(一般为1)

    # 图的总节点数
    n = adj.shape[0]

    # 计算每个节点的度(deg)。由于在coo格式下 row 中存的是边的起点，
    # 我们可以通过对 data 中的值进行聚合得到节点的度
    deg = torch.zeros(n, dtype=torch.float)
    deg.index_add_(0, row, data)  # 将 row[i] 对应节点的度加上 data[i]

    # 计算 deg 的 -1/2 次方
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # 将无穷大替换为0，防止度为0导致除0

    # 对 data 进行 D^-1/2 * data * D^-1/2 的缩放
    # 即边 (u, v) 的权重 = deg_inv_sqrt[u] * w(u,v) * deg_inv_sqrt[v]
    data = deg_inv_sqrt[row] * data * deg_inv_sqrt[col]

    # 用新的data、row、col构造 PyTorch 稀疏矩阵 (n x n)
    A = torch.sparse_coo_tensor(
        indices=torch.stack([row, col], dim=0),
        values=data,
        size=(n, n)
    )
    # coalesce() 会合并重复索引并让 indices 有序，保证稀疏矩阵表示的有效性
    A = A.coalesce()
    return A

##############################################################################
# 定义一个GCNLayer类，来模拟 DGL 中 aggregator_type='gcn' 的 SAGEConv 层逻辑。
# 这里的核心操作是 A * X * W，其中 A 是已经做了归一化的邻接矩阵(A_hat)，
# X是输入特征，W是可学习的权重。
##############################################################################
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None, dropout=0.0):
        super(GCNLayer, self).__init__()
        # 定义可学习参数 W 和 b
        self.weight = nn.Parameter(torch.FloatTensor(in_feats, out_feats))
        self.bias = nn.Parameter(torch.FloatTensor(out_feats))

        # 激活函数可以是relu等，如果为None则不使用激活
        self.activation = activation
        # dropout层，用于特征的随机失活，防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 对参数进行初始化
        nn.init.xavier_uniform_(self.weight)  # Xavier初始化，让权重分布更稳定
        nn.init.zeros_(self.bias)             # 偏置初始化为0

    def forward(self, A, X):
        # X先经过dropout
        X = self.dropout(X)
        # 稀疏乘法：A是 (N x N) 的稀疏矩阵，X是 (N x in_feats)
        # 计算 A * X = (N x in_feats)
        h = torch.sparse.mm(A, X)
        # 线性变换：h * W + b = (N x out_feats)
        h = h @ self.weight + self.bias
        # 如果有激活函数，则进一步处理
        if self.activation is not None:
            h = self.activation(h)
        return h

##############################################################################
# GraphSAGE_model 对应原代码中的 GraphSAGE_model 类，但不再使用 dgl 的 SAGEConv，
# 而是使用上面自定义的 GCNLayer 实现"gcn"汇聚方式。
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
        # 这里我们用 ModuleList 来堆叠多层 GCNLayer
        self.layers = nn.ModuleList()

        # 第一层(输入层)，将 in_feats -> n_hidden
        # 这里的dropout设为0是为了模仿原先的第一层SAGEConv没有feat_drop
        self.layers.append(
            GCNLayer(
                in_feats=in_feats,
                out_feats=n_hidden,
                activation=activation,
                dropout=0.0
            )
        )

        # 如果 n_layers=1，则不会进入循环
        # 若 n_layers>1，则添加中间隐藏层(n_layers-1个)
        for i in range(n_layers - 1):
            self.layers.append(
                GCNLayer(
                    in_feats=n_hidden,
                    out_feats=n_hidden,
                    activation=activation,
                    dropout=dropout
                )
            )

        # 输出层，将 n_hidden -> n_classes，不带激活函数
        self.layers.append(
            GCNLayer(
                in_feats=n_hidden,
                out_feats=n_classes,
                activation=None,
                dropout=dropout
            )
        )

    def forward(self, A, features):
        # 特征初始输入
        h = features
        # 依次经过每一层GCNLayer
        for layer in self.layers:
            h = layer(A, h)
        # 返回最终输出
        return h

##############################################################################
# GraphSAGE 主体类，负责数据加载、模型训练、评估等功能
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
        # 记录开始时间，用于统计训练耗时
        self.t = time.time()

        # 学习率
        self.lr = lr
        # 权重衰减系数
        self.weight_decay = weight_decay
        # 训练迭代次数
        self.epochs = epochs
        # 是否打印训练过程
        self.print_progress = print_progress
        # 是否进行dropEdge以及dropEdge的比例
        self.dropedge = dropedge

        # 设备选择：如果GPU可用且cuda>=0，则使用cuda，否则使用cpu
        if not torch.cuda.is_available():
            cuda = -1
        self.device = torch.device(f'cuda:{cuda%8}' if cuda >= 0 else 'cpu')

        # 随机种子固定(若seed>0)
        if seed > 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # 加载/处理数据
        self.load_data(adj, adj_eval, features, labels, tvt_nids)

        # 初始化模型
        self.model = GraphSAGE_model(
            in_feats=self.features.size(1),
            n_hidden=hidden_size,
            n_classes=self.n_class,
            n_layers=n_layers,
            activation=F.relu,
            dropout=dropout,
            aggregator_type='gcn'  # 这里可随意设置，但我们目前只实现gcn方式
        )
        # 将模型放到指定设备上
        self.model.to(self.device)

    def load_data(self, adj, adj_eval, features, labels, tvt_nids):
        # 将 features 转成 FloatTensor
        if isinstance(features, torch.FloatTensor) or isinstance(features, torch.cuda.FloatTensor):
            self.features = features
        else:
            self.features = torch.FloatTensor(features)

        # 不再对特征维度 (1433, 3703) 进行 L1 归一化

        # 将标签转成对应的tensor
        if len(labels.shape) == 2:
            labels = torch.FloatTensor(labels)
        else:
            labels = torch.LongTensor(labels)
        self.labels = labels

        # 判断是多标签还是单标签，确定类别数
        if len(self.labels.size()) == 1:
            self.n_class = len(torch.unique(self.labels))
        else:
            self.n_class = labels.size(1)

        # 划分训练/验证/测试的节点id
        self.train_nid = tvt_nids['train']
        self.val_nid = tvt_nids['val']
        self.test_nid = tvt_nids['test']

        # -------------------------------------
        # 处理训练用的邻接矩阵
        # -------------------------------------
        assert sp.issparse(adj), "adj 必须是稀疏矩阵格式"
        if not isinstance(adj, sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        # 添加自环
        adj.setdiag(1)
        # 转成csr方便后续处理
        adj = sp.csr_matrix(adj)
        # 保存原始的稀疏矩阵(训练时可能要dropEdge，所以留着原始adj)
        self.adj = adj
        # 先做一次标准化，然后存成 PyTorch sparse tensor
        self.A = normalize_adj(adj)
        # 放到指定设备
        self.A = self.A.to(self.device)

        # -------------------------------------
        # 处理推断/验证用的邻接矩阵
        # -------------------------------------
        assert sp.issparse(adj_eval), "adj_eval 必须是稀疏矩阵格式"
        if not isinstance(adj_eval, sp.coo_matrix):
            adj_eval = sp.coo_matrix(adj_eval)
        # 添加自环
        adj_eval.setdiag(1)
        # 转成csr
        adj_eval = sp.csr_matrix(adj_eval)
        self.adj_eval = adj_eval
        # 标准化后保存
        self.A_eval = normalize_adj(adj_eval)
        self.A_eval = self.A_eval.to(self.device)

    def dropEdge(self):
        # 在原始的 adj(上三角)中随机去掉一部分边
        # 目的是防止过拟合，类似dropout但作用在图的边上

        # 提取上三角部分
        upper = sp.triu(self.adj, 1)
        # 总边数(不包含对角线和下三角重复部分)
        n_edge = upper.nnz
        # 剩余边数
        n_edge_left = int((1 - self.dropedge) * n_edge)

        # 随机选择保留的边(索引)
        index_edge_left = np.random.choice(n_edge, n_edge_left, replace=False)

        # 根据这些索引取出对应的行、列、权重
        data = upper.data[index_edge_left]
        row = upper.row[index_edge_left]
        col = upper.col[index_edge_left]

        # 构造新的上三角的稀疏矩阵
        adj = sp.coo_matrix((data, (row, col)), shape=self.adj.shape)
        # 对称一下，得到完整邻接矩阵
        adj = adj + adj.T
        # 添加自环
        adj.setdiag(1)

        # 对新的adj进行标准化，保存到 self.A 中(专用于训练)
        self.A = normalize_adj(adj)
        self.A = self.A.to(self.device)

    def fit(self):
        # 定义优化器，使用Adam，并带有L2正则(weight_decay)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # 将特征和标签移动到计算设备上
        features = self.features.to(self.device)
        labels = self.labels.to(self.device)

        # 单标签分类，使用 CrossEntropyLoss
        nc_criterion = nn.CrossEntropyLoss()

        # 用于记录最佳验证集精度
        best_vali_acc = 0.0

        # 存储最佳测试集上四项指标
        best_test_acc = 0.0
        best_test_prec = 0.0
        best_test_rec = 0.0
        best_test_f1 = 0.0

        # 记录最佳 epoch
        best_epoch = 0

        # best_logits 用于可视化或分析，但这里主要用来在验证集最优时保存
        best_logits = None

        # 开始迭代训练
        for epoch in range(self.epochs):
            # 如果设置了dropedge>0，就重新drop一次
            if self.dropedge > 0:
                self.dropEdge()

            # 进入训练模式
            self.model.train()

            # 前向传播：计算当前模型输出的logits
            logits = self.model(self.A, features)

            # 计算当前的训练loss(只在训练集节点上)
            loss_train = nc_criterion(logits[self.train_nid], labels[self.train_nid])

            # 反向传播并更新参数
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            # 切换到评估模式，不启用dropout
            self.model.eval()
            with torch.no_grad():
                # 使用完整的评估邻接矩阵A_eval来得到logits
                logits_eval = self.model(self.A_eval, features).detach().cpu()

            # 在验证集上计算评价指标(主要看acc来决定是否更新最优)
            vali_acc, vali_prec, vali_rec, vali_f1 = self.eval_node_cls(
                logits_eval[self.val_nid],
                labels[self.val_nid].cpu()
            )

            # 若需要打印训练过程信息
            if self.print_progress:
                print(f"Epoch [{epoch+1:2d}/{self.epochs}]: loss: {loss_train.item():.4f}, "
                      f"vali acc: {vali_acc:.4f}, vali_prec: {vali_prec:.4f}, vali_rec: {vali_rec:.4f}, vali_f1: {vali_f1:.4f}")

            # 如果验证集上acc更高，就更新最优记录，并在测试集上评估
            if vali_acc > best_vali_acc:
                best_vali_acc = vali_acc
                best_logits = logits_eval
                best_epoch = epoch

                # 在测试集上计算四项指标
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

        # 打印最终测试结果(基于验证集最优时)
        if self.print_progress:
            print("Final test results: "
                  f"acc: {best_test_acc:.4f}, prec: {best_test_prec:.4f}, "
                  f"rec: {best_test_rec:.4f}, f1: {best_test_f1:.4f}")

        # 释放缓存
        del self.model, features, labels, self.A
        torch.cuda.empty_cache()
        gc.collect()

        # 计算训练总耗时
        t_used = time.time() - self.t

        # 返回四项指标：acc, prec, rec, f1 以及最佳 epoch
        return best_test_acc, best_test_prec, best_test_rec, best_test_f1, best_epoch

    def eval_node_cls(self, logits, labels):
        # 单标签分类：直接 argmax
        preds = torch.argmax(logits, dim=1)
        average_type = 'macro'

        # 转成numpy，方便使用sklearn
        preds_np = preds.numpy()
        labels_np = labels.numpy()

        # 计算四项指标
        acc = accuracy_score(labels_np, preds_np)
        prec = precision_score(labels_np, preds_np, average=average_type, zero_division=0)
        rec = recall_score(labels_np, preds_np, average=average_type, zero_division=0)
        f1 = f1_score(labels_np, preds_np, average=average_type, zero_division=0)

        # 返回四项指标
        return acc, prec, rec, f1
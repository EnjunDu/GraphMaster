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

# ★ sklearn.metrics 用于计算acc, prec, rec, f1
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


###############################################################################
# ☆ 自定义的 GraphSAGE 卷积层，用以取代 dgl.nn.pytorch.conv.SAGEConv
#   我们只实现 aggregator_type='gcn' 的聚合方式
###############################################################################
class GraphSAGELayer(nn.Module):
    def __init__(self, in_feats, out_feats, aggregator_type='gcn',
                 feat_drop=0.0, activation=None):
        super(GraphSAGELayer, self).__init__()
        # 线性变换，用于将聚合后的特征映射到 out_feats 维度
        self.fc = nn.Linear(in_feats, out_feats, bias=True)
        # 特征的dropout层
        self.feat_drop = nn.Dropout(feat_drop)
        # 激活函数
        self.activation = activation
        # 聚合器类型，这里只实现 'gcn'
        self.aggregator_type = aggregator_type

    def forward(self, adj_sp, h):
        """
        adj_sp: 传入的是归一化后的图邻接矩阵 (torch.sparse_coo_tensor 或者其他稀疏格式)
        h     : 节点特征，大小 (N, in_feats)
        """
        # 先对特征进行 dropout
        h = self.feat_drop(h)
        # 根据 aggregator_type='gcn'，我们执行: h_agg = A_norm * h
        # 其中 A_norm 是对称归一化的邻接矩阵
        h_agg = torch.sparse.mm(adj_sp, h)

        # 线性变换
        h_agg = self.fc(h_agg)

        # 如果设置了激活函数，则执行
        if self.activation is not None:
            h_agg = self.activation(h_agg)

        return h_agg


###############################################################################
# ☆ Jumping Knowledge Network (JKNet) 模型的主体
#   由若干层 GraphSAGEConv (这里用 GraphSAGELayer 代替) 进行堆叠
#   并在输出时将各层的结果做拼接 (concat) 最终再接一个线性层
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
        # 用于存储多层 GraphSAGE 卷积
        self.layers = nn.ModuleList()

        # 第一层：feat_drop设为0是参考了原代码的写法
        self.layers.append(
            GraphSAGELayer(in_feats, n_hidden, aggregator_type=aggregator_type,
                           feat_drop=0.0, activation=activation)
        )

        # 中间若干层：feat_drop设为 dropout
        for i in range(n_layers - 1):
            self.layers.append(
                GraphSAGELayer(n_hidden, n_hidden, aggregator_type=aggregator_type,
                               feat_drop=dropout, activation=activation)
            )

        # 最终拼接后 (n_layers 个输出拼接)，再映射到 n_classes 维
        self.layer_output = nn.Linear(n_hidden * n_layers, n_classes)

    def forward(self, adj_sp, features):
        # h 用来迭代地存放当前层的特征
        h = features
        # hs 用来收集每一层的特征，以便最后做拼接
        hs = []

        # 逐层做 GraphSAGE 聚合
        for layer in self.layers:
            h = layer(adj_sp, h)
            hs.append(h)

        # 将所有层的输出特征在特征维度上拼接
        h_cat = torch.cat(hs, dim=1)

        # 通过最终线性层得到分类结果
        out = self.layer_output(h_cat)
        return out


###############################################################################
# ☆ JKNet 主类，管理数据、模型构造及训练流程
###############################################################################
class JKNet():
    def __init__(self,
                 adj,          # 训练用的邻接矩阵 (scipy稀疏矩阵)
                 adj_eval,     # 推断/评估用的邻接矩阵 (scipy稀疏矩阵)
                 features,     # 节点特征
                 labels,       # 节点标签
                 tvt_nids,     # 训练/验证/测试节点索引
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
        # 初始化起始时间，用于观察程序耗时
        self.t = time.time()
        # 学习率
        self.lr = lr
        # 权重衰减系数
        self.weight_decay = weight_decay
        # 训练轮数
        self.epochs = epochs
        # 是否打印训练进度
        self.print_progress = print_progress
        # dropedge 比例
        self.dropedge = dropedge

        # 配置设备，若没有可用GPU则使用CPU
        if not torch.cuda.is_available():
            cuda = -1
        self.device = torch.device(f'cuda:{cuda % 8}' if cuda >= 0 else 'cpu')

        # 固定随机种子，保证可复现 (若 seed>0)
        if seed > 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # 加载数据并做相应初始化处理
        self.load_data(adj, adj_eval, features, labels, tvt_nids)

        # 构建模型
        self.model = JKNet_model(
            in_feats=self.features.size(1), # 输入维度 = 特征维度
            n_hidden=hidden_size,           # 隐藏层大小
            n_classes=self.n_class,         # 类别数量
            n_layers=n_layers,             # 层数
            activation=F.relu,             # 激活函数
            dropout=dropout,               # dropout
            aggregator_type='gcn'          # 默认gcn聚合方式
        )
        # 模型放到指定设备上
        self.model.to(self.device)


    ############################################################################
    # ☆ 将 scipy 的稀疏矩阵转换为 pytorch 的稀疏矩阵，并做对称归一化
    ############################################################################
    def normalize_adj_torch(self, adj_sp):
        """
        输入:  adj_sp (scipy.sparse), 形状 (N, N)，假定已经含有自环 (对角线=1)
        输出:  adj_norm_torch (torch.sparse), 形状 (N, N)
        """
        # 将adj_sp先转换为 CSR格式，便于行访问
        adj_csr = sp.csr_matrix(adj_sp)
        # 计算度
        deg = np.array(adj_csr.sum(1))  # shape = (N,1)
        # 计算 D^-1/2
        d_inv_sqrt = np.power(deg, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

        # D^-1/2 放进对角阵
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        # 计算对称归一化 A_norm = D^-1/2 * A * D^-1/2
        adj_norm = d_inv_sqrt_mat.dot(adj_csr).dot(d_inv_sqrt_mat)

        # 转换为COO格式便于构建 torch.sparse
        adj_coo = sp.coo_matrix(adj_norm)
        # 构造 indices
        indices = torch.LongTensor([adj_coo.row, adj_coo.col])
        # 构造 values
        values = torch.FloatTensor(adj_coo.data)
        # 形状
        shape = adj_coo.shape

        # 用 torch.sparse_coo_tensor 建立稀疏张量
        adj_norm_torch = torch.sparse_coo_tensor(indices, values, torch.Size(shape))
        return adj_norm_torch


    ############################################################################
    # ☆ 加载数据，并为训练与评估分别准备好邻接矩阵
    ############################################################################
    def load_data(self, adj, adj_eval, features, labels, tvt_nids):
        # 将特征转为 FloatTensor
        if isinstance(features, torch.FloatTensor) or isinstance(features, torch.cuda.FloatTensor):
            self.features = features
        else:
            self.features = torch.FloatTensor(features)

        # 不再对特征做 L1 归一化

        # 将标签转为 Tensor
        if len(labels.shape) == 2:
            labels = torch.FloatTensor(labels)
        else:
            labels = torch.LongTensor(labels)
        self.labels = labels

        # 若标签是一维（多分类），则类别数 = unique(labels) 的数量
        # 若标签是二维（多标签），则类别数 = label.size(1)
        if len(self.labels.size()) == 1:
            self.n_class = len(torch.unique(self.labels))
        else:
            self.n_class = self.labels.size(1)

        # 取出训练、验证、测试的节点编号 (从字典中获取)
        self.train_nid = tvt_nids['train']
        self.val_nid = tvt_nids['val']
        self.test_nid = tvt_nids['test']

        # 训练用的邻接矩阵 (带自环)
        assert sp.issparse(adj)
        if not isinstance(adj, sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        adj.setdiag(1)
        self.adj = adj  # scipy coo_matrix

        # 推断/评估用的邻接矩阵 (带自环)
        assert sp.issparse(adj_eval)
        if not isinstance(adj_eval, sp.coo_matrix):
            adj_eval = sp.coo_matrix(adj_eval)
        adj_eval.setdiag(1)
        self.adj_eval = adj_eval  # scipy coo_matrix

        # 预先做评估用邻接矩阵的归一化，存为 self.adj_eval_norm
        self.adj_eval_norm = self.normalize_adj_torch(self.adj_eval).to(self.device)
        # 若训练时不dropedge，就直接也把 self.adj_norm 准备好
        if self.dropedge <= 0:
            self.adj_norm = self.normalize_adj_torch(self.adj).to(self.device)


    ############################################################################
    # ☆ 随机丢弃部分边，以得到dropedge后的邻接矩阵，并存为 self.adj_norm
    ############################################################################
    def dropEdge(self):
        # 先取上三角
        upper = sp.triu(self.adj, 1)
        # 原本的边数 (不含对角线)
        n_edge = upper.nnz
        # 保留下来的边数
        n_edge_left = int((1 - self.dropedge) * n_edge)
        # 随机选出 n_edge_left 条边保留
        index_edge_left = np.random.choice(n_edge, n_edge_left, replace=False)
        # 得到保留下的边
        data = upper.data[index_edge_left]
        row = upper.row[index_edge_left]
        col = upper.col[index_edge_left]
        # 重新构建邻接矩阵的上三角
        adj = sp.coo_matrix((data, (row, col)), shape=self.adj.shape)
        # 对称化
        adj = adj + adj.T
        # 加上自环
        adj.setdiag(1)

        # 调用 normalize_adj_torch 做归一化
        self.adj_norm = self.normalize_adj_torch(adj).to(self.device)


    ############################################################################
    # ☆ 模型训练函数
    #   需要返回：best_test_acc, best_test_prec, best_test_rec, best_test_f1, best_epoch
    ############################################################################
    def fit(self):
        # 定义优化器 (Adam)
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        # 将 features, labels 放到 device 上
        features = self.features.to(self.device)
        labels = self.labels.to(self.device)

        # 使用 CrossEntropyLoss 作为损失函数 (单标签任务)
        nc_criterion = nn.CrossEntropyLoss()

        # 用于记录最佳验证 micro-f1
        best_vali_f1 = 0.0
        # 用于保存对应的测试集四项指标
        best_test_acc = 0.0
        best_test_prec = 0.0
        best_test_rec = 0.0
        best_test_f1 = 0.0
        # 记录最佳 epoch
        best_epoch = 0

        # 训练若干轮
        for epoch in range(self.epochs):
            # 若需要dropedge，每个 epoch 都要重新随机丢边
            if self.dropedge > 0:
                self.dropEdge()
            # 否则用预先存好的 adj_norm
            else:
                pass

            # 训练模式
            self.model.train()
            # 前向传播 (训练邻接使用 self.adj_norm)
            logits = self.model(self.adj_norm, features)
            # 计算训练集上的损失 (只对 train_nid 计算)
            loss_train = nc_criterion(logits[self.train_nid], labels[self.train_nid])

            # 反向传播
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            # 验证：这里不需要dropout等，设置 model.eval()
            self.model.eval()
            with torch.no_grad():
                # 评估用邻接 (self.adj_eval_norm)，前向传播
                logits_eval = self.model(self.adj_eval_norm, features)
            # 计算验证集的指标
            vali_acc, vali_prec, vali_rec, vali_f1 = self.eval_node_cls(
                logits_eval[self.val_nid], labels[self.val_nid]
            )

            if self.print_progress:
                print(f"Epoch [{epoch+1:2d}/{self.epochs}]: loss: {loss_train.item():.4f}, "
                      f"vali_acc: {vali_acc:.4f}, vali_prec: {vali_prec:.4f}, vali_rec: {vali_rec:.4f}, vali_f1: {vali_f1:.4f}")

            # 若验证集 f1 表现更好，则在测试集上评估并更新最佳指标
            if vali_f1 > best_vali_f1:
                best_vali_f1 = vali_f1
                best_epoch = epoch
                # 在测试集上计算四项指标
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

        # 训练完成后，打印最终结果
        if self.print_progress:
            print("Final test results:")
            print(f"acc: {best_test_acc:.4f}, prec: {best_test_prec:.4f}, "
                  f"rec: {best_test_rec:.4f}, f1: {best_test_f1:.4f}")

        # 释放 GPU 显存
        del self.model, features, labels, self.adj_norm, self.adj_eval_norm
        torch.cuda.empty_cache()
        gc.collect()

        # 返回最佳测试集指标和最佳 epoch
        return best_test_acc, best_test_prec, best_test_rec, best_test_f1, best_epoch


    ############################################################################
    # ☆ 评估节点分类指标：acc, prec, rec, f1
    ############################################################################
    def eval_node_cls(self, logits, labels):
        """
        logits: 维度 (N, n_class)，预测的原始值
        labels: 维度 (N,)，真实标签
        """
        # 单标签分类：取最大值索引
        preds = torch.argmax(logits, dim=1)

        # 转到 CPU，转成 numpy，以便用 sklearn 度量
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()

        # 计算四项指标
        acc = accuracy_score(labels_np, preds_np)
        prec = precision_score(labels_np, preds_np, average='macro', zero_division=0)
        rec = recall_score(labels_np, preds_np, average='macro', zero_division=0)
        f1 = f1_score(labels_np, preds_np, average='macro', zero_division=0)

        return acc, prec, rec, f1
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

class HierarchicalModularReservoir:
    def __init__(self, config):
        """
        初始化支持 TRNN 动力学的储备池。
        """
        np.random.seed(config.seed)
        self.config = config
        self.N = config.N
        self._build_network()

    def _make_adj_hierarchical_modular(self):
        """创建分层模块化稀疏矩阵 (保持你原有的逻辑，稍作优化)"""
        N = self.N
        k_target = self.config.k_target
        n_modules_per_level = self.config.n_modules
        
        if not n_modules_per_level:
            # 计算连接概率 p = k / N
            p = k_target / N
            # 使用 networkx 生成随机图，或者直接生成随机矩阵
            # 这里为了保持一致性，生成一个随机稀疏矩阵
            # print(f"No modules defined. Generating standard random matrix with p={p:.4f}")
            # 生成随机掩码：值小于 p 的位置为 1，否则为 0
            mask = (np.random.rand(N, N) < p).astype(float)
            np.fill_diagonal(mask, 0)
            return csr_matrix(mask)
        
        A = np.zeros((N, N))
        for level_idx, n_mods in enumerate(n_modules_per_level):
            module_size = N // n_mods
            sizes = [module_size] * n_mods
            
            # 简单的概率分配逻辑
            base_p = k_target / N
            p_in = base_p * (2.0 if level_idx == 0 else 1.2)
            p_out = base_p * 0.2
            
            probs = [[p_in if i == j else p_out for j in range(n_mods)] for i in range(n_mods)]
            G_level = nx.stochastic_block_model(sizes, probs, directed=True)
            A += nx.to_numpy_array(G_level)
            
        np.fill_diagonal(A, 0)
        return csr_matrix(A)

    def _build_network(self):
        # 1. 拓扑结构
        A_sparse = self._make_adj_hierarchical_modular()
        
        # 2. 循环权重 W_res
        # 论文中使用了Dale法则(E/I分离)，这里简化为正态分布，但引入稀疏性
        W_res_dense = A_sparse.toarray() * np.random.randn(self.N, self.N)
        
        # 调整光谱半径
        eigenvalues = np.linalg.eigvals(W_res_dense)
        current_rho = np.max(np.abs(eigenvalues))
        if current_rho > 0:
            W_res_dense = (W_res_dense / current_rho) * self.config.rho
        self.W_res = W_res_dense

        # 3. 输入权重 W_in [修改点 1]
        # 形状必须是 (N_neurons, N_input)，即 (600, 24)
        # 这样才能将 24维输入投影到 600维的储备池空间
        self.W_in = (np.random.rand(self.N, self.config.N_input) * 2 - 1) * self.config.W_in_scale

    def run(self, u_seq, washout):
        """
        运行模型，实现 TRNN 动力学方程 (Eq. 6 & 7 in paper).
        r_t = (1 - alpha_r) * r_{t-1} + alpha_r * (ReLU(Input + Recurrent) - gamma * V_{t-1})
        V_t = (1 - alpha_v) * V_{t-1} + alpha_v * (m * r_{t-1})
        """
        T = len(u_seq)
        N = self.N
        cfg = self.config
        
        # 初始化状态
        r = np.zeros(N) # 神经元放电率
        v = np.zeros(N) # 适应性变量 (自抑制)
        
        X_all = np.zeros((T, N))

        for t in range(T):
            # [修改点 2] 计算突触输入
            # u_seq[t] shape is (24,)
            # self.W_in shape is (600, 24)
            # self.W_in @ u_seq[t] resulting shape is (600,)
            input_current = self.W_in @ u_seq[t]
            synaptic_input = self.W_res @ r + input_current + np.random.normal(0, 0.1, N) # 加入噪声
            
            # TRNN 核心更新公式
            # 1. 更新放电率 r (包含 -gamma * v 的抑制项)
            activation = np.maximum(0, synaptic_input) # ReLU
            r_new = (1 - cfg.alpha_r) * r + cfg.alpha_r * (activation - cfg.gamma * v)
            r_new = np.maximum(0, r_new) # 确保放电率非负
            
            # 2. 更新适应性变量 v
            v_new = (1 - cfg.alpha_v) * v + cfg.alpha_v * (cfg.m * r)
            
            # 更新状态
            r = r_new
            v = v_new
            
            X_all[t] = r
            # # [关键修正] 
            # # 适应性变量 v 作为抑制性电流，在非线性激活之前减去。
            # # 这允许强输入能够克服弱抑制，而不是被强制减法归零。
            # effective_input = synaptic_input - cfg.gamma * v
            
            # # 2. 计算目标放电率 (Target Firing Rate)
            # firing_target = np.maximum(0, effective_input) # ReLU
            
            # # 3. 泄漏积分 (Leaky Integration) 更新 r
            # # r[t] = (1-alpha) * r[t-1] + alpha * target
            # r_new = (1 - cfg.alpha_r) * r + cfg.alpha_r * firing_target
            
            # # 4. 更新适应性变量 v
            # v_new = (1 - cfg.alpha_v) * v + cfg.alpha_v * (cfg.m * r)
            
            # r = r_new
            # v = v_new
            
            # X_all[t] = r

        return X_all[washout:]
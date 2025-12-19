import os
import numpy as np

# --- Directory Settings ---
CHECKPOINT_DIR = "checkpoints"
VISUALIZE_DIR = "visualize"
TI_HIGH_DIR = os.path.join(CHECKPOINT_DIR, "TI_H")
TI_LOW_DIR = os.path.join(CHECKPOINT_DIR, "TI_L")

os.makedirs(TI_HIGH_DIR, exist_ok=True)
os.makedirs(TI_LOW_DIR, exist_ok=True)
os.makedirs(VISUALIZE_DIR, exist_ok=True)

# --- Model Parameters ---
N = 600  # 稍微增加神经元数量以获得更好的高维投影
# dims = [200, 200, 200] # Sensor, Association, Motor
k_target = 6
n_modules = []
n_levels = len(n_modules)
rho = 1.2
# 输入采用Von Mises 调谐曲线，以确保输入在高维空间尽可能保持方向完全不同（甚至正交）的向量
# 增加其可分离性，模型可以更好的在高维空间决策

N_input = 24        # 论文设定为 24 个输入神经元
W_in_scale = 1.0    # 保持不变，用于缩放最终输入

# [新增] Von Mises 编码参数 (参考论文 Eq. 1)
VM_KAPPA = 4.0      # 浓度系数
VM_AMPLITUDE = 1.0  # 幅度 A (论文中是 4/exp(k)，这里简化为1，由 W_in_scale 控制最终强度)
INPUT_NOISE = 0.05  # 适当加一点输入噪声增加鲁棒性

# --- TRNN Specific Parameters (基于论文 Methods) ---
# 论文中: alpha_r=0.6 (ODPA), alpha_v=0.1, gamma=2, m=2
dt = 0.1          # 模拟时间步长（ms）
alpha_r = 0.1     # 神经元更新率
alpha_v = 0.16     # 适应性变量更新率
gamma = 0.0       # 自抑制强度 (关键参数!)
m = 2.0           # 适应性积累速率

seed = 42

# --- Task Parameters ---
odors = ['A', 'B']
delay_steps_short = 30 # 3s (假设 dt=100ms)
delay_steps_long = 60  # 6s
trial_count_train = 300
trial_count_test = 100
T_washout = 20

# --- Training Parameters ---
REG = 1e-4
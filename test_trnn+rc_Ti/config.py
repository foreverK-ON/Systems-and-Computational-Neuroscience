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
N = 500  # 稍微增加神经元数量以获得更好的高维投影
k_target = 10
n_modules = [10, 5] 
n_levels = len(n_modules)
W_in_scale = 1.0  # 增加输入强度
rho = 1.2         # 光谱半径

# --- TRNN Specific Parameters (基于论文 Methods) ---
# 论文中: alpha_r=0.6 (ODPA), alpha_v=0.1, gamma=2, m=2
dt = 0.1          # 模拟时间步长
tau_r = 0.1 / 0.6 # 推导出的时间常数
tau_v = 1.0       # 适应性变量时间常数通常较慢
alpha_r = 0.6     # 神经元更新率
alpha_v = 0.1     # 适应性变量更新率
gamma = 2.0       # 自抑制强度 (关键参数!)
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
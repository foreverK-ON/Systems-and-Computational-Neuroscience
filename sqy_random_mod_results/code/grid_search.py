import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import time
import json

# 导入现有模块
import config as base_config
from model import HierarchicalModularReservoir
from task_odpa import generate_odpa_sequence, run_model_on_odpa_trials, evaluate_accuracy

# ==========================================
# 1. 辅助类与函数
# ==========================================

class DynamicConfig:
    """用于动态覆盖原始 config 模块参数的类"""
    def __init__(self, overrides):
        # 复制 base_config 的所有属性
        for key in dir(base_config):
            if not key.startswith("__"):
                setattr(self, key, getattr(base_config, key))
        
        # 应用覆盖
        for key, value in overrides.items():
            setattr(self, key, value)

def train_readout_on_response_only(X_states_list, targets_list, infos, reg=1e-4):
    """
    (复用修正后的训练逻辑) 仅使用 Response 阶段训练
    """
    X_train_filtered = []
    Y_train_filtered = []

    for X, Y, info in zip(X_states_list, targets_list, infos):
        start, end = info['indices']['resp']
        # 简单起见，取 Response 阶段的全部
        X_resp = X[start:end]
        Y_resp = Y[start:end]
        
        X_train_filtered.append(X_resp)
        Y_train_filtered.append(Y_resp)

    if not X_train_filtered:
        return np.zeros((X_states_list[0].shape[1], 3))

    X_train_concat = np.vstack(X_train_filtered)
    Y_train_concat = np.vstack(Y_train_filtered)
    
    N = X_train_concat.shape[1]
    I = np.eye(N)
    # Ridge Regression
    try:
        W_out = np.linalg.solve(X_train_concat.T @ X_train_concat + reg * I, X_train_concat.T @ Y_train_concat)
    except np.linalg.LinAlgError:
        W_out = np.zeros((N, 3)) # 避免奇异矩阵崩溃
        
    return W_out

def evaluate_single_config(params):
    """
    运行单个配置的评估任务。
    此函数将在多进程中运行。
    """
    # 解包参数
    rho, gamma, alpha_r, run_id = params
    
    # 覆盖配置
    overrides = {
        'rho': rho,
        'gamma': gamma,
        'alpha_r': alpha_r,
        'seed': 42 + run_id, # 确保每次运行略有不同，或者固定以便比较
        # 强制设置一些确保任务难度的参数
        'delay_steps_long': 60, # 6秒长延迟--先测试一下短延迟
        'trial_count_train': 400,
        'trial_count_test': 100,
        'VM_AMPLITUDE': 3.0,
        'W_in_scale': 1.0
    }
    cfg = DynamicConfig(overrides)
    
    try:
        # 1. 初始化模型
        model = HierarchicalModularReservoir(cfg)
        
        # 2. 生成训练数据 (短延迟用于训练，或者长短混合，这里用短延迟训练看泛化)
        # 为了更严格，我们直接用长延迟训练，或者用短延迟训练测试长延迟
        # 这里采用：用 Short 训练，测 Long (测试泛化和记忆极限)
        train_inputs, train_targets, train_infos = generate_odpa_sequence(
            cfg.odors, cfg.delay_steps_short, cfg.trial_count_train)
        
        # 3. 运行训练
        X_train = run_model_on_odpa_trials(model, train_inputs, washout_steps=0)
        W_out = train_readout_on_response_only(X_train, train_targets, train_infos, reg=cfg.REG)
        
        # 4. 生成测试数据 (Long Delay)
        test_inputs, test_targets, test_infos = generate_odpa_sequence(
            cfg.odors, cfg.delay_steps_long, cfg.trial_count_test)
        
        # 5. 运行测试
        X_test = run_model_on_odpa_trials(model, test_inputs, washout_steps=0)
        acc_long = evaluate_accuracy(X_test, W_out, test_infos)
        
        return {
            'rho': rho,
            'gamma': gamma,
            'alpha_r': alpha_r,
            'accuracy': acc_long
        }
        
    except Exception as e:
        print(f"Error with params {params}: {e}")
        return {
            'rho': rho,
            'gamma': gamma,
            'alpha_r': alpha_r,
            'accuracy': 0.0
        }

# ==========================================
# 2. 网格搜索主逻辑
# ==========================================

def run_grid_search():
    print("=== Starting Grid Search for ODPA Task (Long Delay) ===")
    
    # --- 定义搜索空间 ---
    # 根据之前的分析，rho 需要接近或大于 1，gamma 需要较小，alpha_r 需要较小
    rhos = [0.8, 0.9, 0.95,0.98, 
            1.0, 1.2]
    gammas = [0.0,0.2,0.5,0.8,1.0,1.2,1.5]
    alpha_rs = [0.1,0.2,0.3,0.4,0.5] # 也可以加入 0.3
    
    # 生成参数组合
    # 格式: (rho, gamma, alpha_r, run_id)
    # 每个组合我们可以跑几次取平均，这里演示每个组合跑 1 次 (run_id=0)
    param_grid = list(itertools.product(rhos, gammas, alpha_rs, [0]))
    
    print(f"Total configurations to test: {len(param_grid)}")
    
    start_time = time.time()
    
    results = []
    # 使用多进程并行计算 (根据你的CPU核数调整 max_workers)
    # 注意：如果内存不足，请减少 max_workers
    with ProcessPoolExecutor(max_workers=8) as executor:
        for i, res in enumerate(executor.map(evaluate_single_config, param_grid)):
            results.append(res)
            if (i + 1) % 5 == 0:
                print(f"Completed {i + 1}/{len(param_grid)} runs...")

    end_time = time.time()
    print(f"Grid Search Complete in {end_time - start_time:.2f} seconds.")
    
    # --- 处理结果 ---
    df = pd.DataFrame(results)
    
    # 找到最佳参数
    best_row = df.loc[df['accuracy'].idxmax()]
    print("\n=== Best Configuration ===")
    print(best_row)
    
    # 保存结果
    df.to_csv("grid_search_results.csv", index=False)
    print("Results saved to grid_search_results.csv")
    
    return df

# ==========================================
# 3. 可视化
# ==========================================

def plot_heatmap(df, fixed_alpha_r):
    """
    绘制给定 alpha_r 下 rho 和 gamma 的热力图
    """
    subset = df[df['alpha_r'] == fixed_alpha_r]
    if subset.empty:
        return
    
    pivot_table = subset.pivot(index="gamma", columns="rho", values="accuracy")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="viridis", vmin=0, vmax=100)
    plt.title(f"Long Delay Accuracy (alpha_r={fixed_alpha_r})")
    plt.ylabel("Inhibition Gamma")
    plt.xlabel("Spectral Radius Rho")
    plt.savefig(f"grid_search_heatmap_alpha{fixed_alpha_r}.png")
    plt.show()

if __name__ == "__main__":
    # 确保在 Windows 下多进程正常工作
    df = run_grid_search()
    
    # 为每个 alpha_r 绘制一张热力图
    unique_alphas = df['alpha_r'].unique()
    for alpha in unique_alphas:
        plot_heatmap(df, alpha)
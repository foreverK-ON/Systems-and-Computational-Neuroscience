import numpy as np
import config
from model import HierarchicalModularReservoir
from task_odpa import generate_odpa_sequence, run_model_on_odpa_trials, evaluate_accuracy
from utils import calculate_ti_for_trials, plot_paper_style_heatmap, identify_selective_neurons,plot_reservoir_connectivity

def train_readout_on_response_only(X_states_list, targets_list, infos, reg=1e-4):
    """
    仅使用 Response (Test) 阶段的数据来训练读出层。
    这样可以防止 Fixation (大量 0 标签) 造成的类别不平衡。
    """
    X_train_filtered = []
    Y_train_filtered = []

    for X, Y, info in zip(X_states_list, targets_list, infos):
        # 获取 Response 阶段的索引
        start, end = info['indices']['resp']
        
        # 为了更加稳健，我们可以去掉 Response 的前 1-2 个时间步（瞬态），取中后段
        # 但如果是 Test 即 Response，取全部也可以
        
        # 提取数据
        X_resp = X[start:end]
        Y_resp = Y[start:end]
        
        # 关键：我们只关心 Match (idx 1) 和 Non-Match (idx 2)
        # Fixation (idx 0) 在 Response 阶段应该是 0，所以也可以一起训练
        
        X_train_filtered.append(X_resp)
        Y_train_filtered.append(Y_resp)

    X_train_concat = np.vstack(X_train_filtered)
    Y_train_concat = np.vstack(Y_train_filtered)

    print(f"--- Training Readout on Response Phase (Data shape: {X_train_concat.shape}) ---")
    
    # Ridge Regression
    N = X_train_concat.shape[1]
    I = np.eye(N)
    # W_out = (X^T X + reg I)^-1 X^T Y
    W_out = np.linalg.solve(X_train_concat.T @ X_train_concat + reg * I, X_train_concat.T @ Y_train_concat)
    
    return W_out

def main():
    # 1. 初始化模型 (传入整个 config 对象)
    print(f"--- Creating TRNN Model (gamma={config.gamma}) ---")
    model = HierarchicalModularReservoir(config)

    # 2. 生成数据
    print("--- Generating Training Data (Short Delay) ---")
    train_inputs, train_targets, train_infos = generate_odpa_sequence(
        config.odors, config.delay_steps_short, config.trial_count_train)
    
    # 3. 收集状态用于训练
    print("--- Collecting States ---")
    X_train_list = run_model_on_odpa_trials(model, train_inputs, washout_steps=0)
    
    # 4. 训练 Readout (Ridge Regression)
    # 我们只关心能否预测Match/Non-match，这需要在 Test 期做出决定
    # 4. 训练 Readout (修正版：只训练 Response 阶段)
    W_out = train_readout_on_response_only(X_train_list, train_targets, train_infos, reg=config.REG)
    
    # ==========================================
    # Phase 1: Short Delay Testing
    # ==========================================
    print("\n=== Phase 1: Short Delay Test ===")
    test_inputs, test_targets, test_infos = generate_odpa_sequence(
        config.odors, config.delay_steps_short, config.trial_count_test)
    
    X_test_list = run_model_on_odpa_trials(model, test_inputs, washout_steps=0)
    acc_short = evaluate_accuracy(X_test_list, W_out, test_infos)
    print(f"Short Delay Accuracy: {acc_short:.2f}%")

    # 计算 TI (Short)
    sel_indices_short = identify_selective_neurons(X_test_list, test_infos)
    avg_ti_short, _ = calculate_ti_for_trials(X_test_list, test_infos)
    print(f"Short Delay TI: {avg_ti_short:.4f}")

    # 可视化 (Short)
    if len(X_test_list) > 0 and len(sel_indices_short) > 0:
        plot_paper_style_heatmap(
            X_test_list[0], 
            test_infos[0], 
            sel_indices=sel_indices_short,
            normalize=True,
            ti_score=avg_ti_short,  # <--- 传入 TI
            filename_suffix=f"short_delay_acc{acc_short:.0f}"
        )

    # ==========================================
    # Phase 2: Long Delay Testing (泛化能力)
    # ==========================================
    print("\n=== Phase 2: Long Delay Test (Generalization) ===")
    # 生成长延迟数据
    test_long_inputs, test_long_targets, test_long_infos = generate_odpa_sequence(
        config.odors, config.delay_steps_long, config.trial_count_test)
    
    # 运行模型 (使用相同的 W_out)
    X_test_long_list = run_model_on_odpa_trials(model, test_long_inputs, washout_steps=0)
    acc_long = evaluate_accuracy(X_test_long_list, W_out, test_long_infos)
    print(f"Long Delay Accuracy: {acc_long:.2f}%")

    # 计算 TI (Long) - 需要重新识别长延迟下的选择性神经元
    print("Identifying selective neurons for Long Delay...")
    sel_indices_long = identify_selective_neurons(X_test_long_list, test_long_infos)
    avg_ti_long, _ = calculate_ti_for_trials(X_test_long_list, test_long_infos)
    print(f"Long Delay TI: {avg_ti_long:.4f}")

    # 可视化 (Long)
    if len(X_test_long_list) > 0:
        # 找一个匹配(Match)且预测正确的例子来画图
        plot_idx = 0
        for i, info in enumerate(test_long_infos):
            if info['is_match']: 
                plot_idx = i
                break
        
        plot_paper_style_heatmap(
            X_test_long_list[plot_idx], 
            test_long_infos[plot_idx], 
            sel_indices=sel_indices_long, # 使用长延迟的选择性神经元进行排序
            normalize=True,
            ti_score=avg_ti_long, # <--- 传入 Long Delay TI
            filename_suffix=f"long_delay_acc{acc_long:.0f}"
        )

    # 随机生成模型无需保存，没有意义，直接记住随机种子就可以了

    # ==========================================
    # Phase 4: Visualize Connectivity
    # ==========================================
    suffix = f"shortAcc_{acc_short:.1f}_longAcc_{acc_long:.1f}_longTI_{avg_ti_long:.2f}"
    print("\n=== Visualizing Connectivity ===")
    plot_reservoir_connectivity(model.W_res, filename_suffix=suffix)

    print("\n--- Main Execution Complete ---")
        

if __name__ == "__main__":
    main()
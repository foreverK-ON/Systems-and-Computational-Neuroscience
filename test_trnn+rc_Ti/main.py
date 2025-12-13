import numpy as np
import config
from model import HierarchicalModularReservoir
from task_odpa import generate_odpa_sequence, run_model_on_odpa_trials
from utils import calculate_ti_for_trials, plot_paper_style_heatmap, identify_selective_neurons

def main():
    # 1. 初始化模型 (传入整个 config 对象)
    print(f"--- Creating TRNN Model (gamma={config.gamma}) ---")
    model = HierarchicalModularReservoir(config)

    # 2. 生成数据
    print("--- Generating Data ---")
    train_inputs, train_targets, train_infos = generate_odpa_sequence(
        config.odors, config.delay_steps_short, config.trial_count_train)
    test_inputs, test_targets, test_infos = generate_odpa_sequence(
        config.odors, config.delay_steps_short, config.trial_count_test)

    # 3. 收集状态用于训练
    print("--- Collecting States ---")
    X_train_list = run_model_on_odpa_trials(model, train_inputs, washout_steps=0)
    
    # 4. 训练 Readout (Ridge Regression)
    # 我们只关心能否预测Match/Non-match，这需要在 Test 期做出决定
    # 策略：使用所有时间步进行训练，让 Readout 每一刻都尝试预测当前的目标
    X_train = np.vstack(X_train_list) # (Total_Steps, N)
    Y_train = np.vstack(train_targets) # (Total_Steps, 3)

    print(f"--- Training Readout (Shape: {X_train.shape}) ---")
    # W_out = (X^T X + reg I)^-1 X^T Y
    I = np.eye(config.N)
    W_out = np.linalg.solve(X_train.T @ X_train + config.REG * I, X_train.T @ Y_train)

    # 5. 测试
    print("--- Testing ---")
    X_test_list = run_model_on_odpa_trials(model, test_inputs, washout_steps=0)
    
    correct = 0
    total = len(test_infos)
    
    for i, (X, info) in enumerate(zip(X_test_list, test_infos)):
        # 预测
        Y_pred = X @ W_out
        
        # 评估逻辑：在 Test 期结束前取平均或最后一步
        # Test start: fix + stim + delay
        test_start = info['fix_steps'] + info['stim_steps'] + info['delay_steps']
        test_end = test_start + info['stim_steps']
        
        # 取测试期中间的预测值
        decision_idx = (test_start + test_end) // 2
        pred_vector = Y_pred[decision_idx] # [Fix, Match, Non-Match]
        
        is_match_pred = pred_vector[1] > pred_vector[2]
        if is_match_pred == info['is_match']:
            correct += 1

    acc = correct / total * 100
    print(f"Test Accuracy: {acc:.2f}%")

    # 6. 计算 TI (关键步骤)
    print("--- Calculating TI ---")
    # 先识别选择性神经元
    sel_indices = identify_selective_neurons(X_test_list, test_infos)
    
    # 计算 TI
    avg_ti, _ = calculate_ti_for_trials(X_test_list, test_infos)
    print(f"Average TI: {avg_ti:.4f}")

    # 7. 可视化 (修改部分)
    print("--- Visualizing Neural Activities (Paper Style) ---")
    if len(X_test_list) > 0:
        # 选择一个有趣的试验进行绘制 (例如 Match 且成功的试验)
        trial_idx_to_plot = 0
        
        # 必须先识别选择性神经元 (基于整个测试集)
        # 如果前面 calculate_ti 还没算，这里算一下
        print("Identifying selective neurons for visualization...")
        sel_indices = identify_selective_neurons(X_test_list, test_infos, p_value_threshold=0.05)
        
        if len(sel_indices) < 5:
            print(f"Warning: Only {len(sel_indices)} selective neurons found. Plot might look empty.")
        
        # 绘制短延迟 (Short Delay) 试验
        plot_paper_style_heatmap(
            X_test_list[trial_idx_to_plot], 
            test_infos[trial_idx_to_plot], 
            sel_indices=sel_indices,
            normalize=True, # 开启归一化以获得更好的视觉效果
            filename_suffix=f"short_delay_trial_{test_infos[trial_idx_to_plot]['trial_id']}"
        )

        # # 如果你跑了长延迟 (Long Delay) 测试，也可以画一个
        # if 'all_test_long_X_states' in locals() and len(all_test_long_X_states) > 0:
        #     plot_paper_style_heatmap(
        #         all_test_long_X_states[0], 
        #         test_long_trial_infos[0], 
        #         sel_indices=sel_indices, # 尝试复用短延迟的选择性神经元，或者重新计算
        #         normalize=True,
        #         filename_suffix=f"long_delay_trial_{test_long_trial_infos[0]['trial_id']}"
        #     )

if __name__ == "__main__":
    main()
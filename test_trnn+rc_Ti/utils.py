import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from scipy.stats import entropy, ranksums
from config import TI_HIGH_DIR, TI_LOW_DIR, VISUALIZE_DIR, CHECKPOINT_DIR

def identify_selective_neurons(all_activities, trial_infos, p_value_threshold=0.05):
    """
    筛选记忆选择性神经元 (基于论文逻辑)。
    比较 Sample A 和 Sample B 在延迟期的活动差异。
    """
    delay_activities_A = []
    delay_activities_B = []

    for act, info in zip(all_activities, trial_infos):
        # 提取延迟期平均活动
        start = info['fix_steps'] + info['stim_steps']
        end = start + info['delay_steps']
        mean_delay_act = np.mean(act[start:end, :], axis=0) # (N,)
        
        if info['sample_odor'] == 'A':
            delay_activities_A.append(mean_delay_act)
        elif info['sample_odor'] == 'B':
            delay_activities_B.append(mean_delay_act)

    delay_activities_A = np.array(delay_activities_A) # (Num_A, N)
    delay_activities_B = np.array(delay_activities_B) # (Num_B, N)
    
    selective_indices = []
    N = delay_activities_A.shape[1]
    
    for n in range(N):
        # Wilcoxon rank-sum test
        _, p = ranksums(delay_activities_A[:, n], delay_activities_B[:, n])
        if p < p_value_threshold:
            selective_indices.append(n)
            
    return np.array(selective_indices)

def calculate_ti_single_trial(neural_activity_trial, selective_indices, delay_start, delay_end):
    """只针对选择性神经元计算 TI"""
    if len(selective_indices) == 0:
        return 0.0, 0.0, 0.0, 0.0 # TI, E, R, P

    # 仅提取选择性神经元的数据
    full_activity_sel = neural_activity_trial[:, selective_indices]
    delay_activity_sel = full_activity_sel[delay_start:delay_end, :] # (T_delay, N_sel)
    
    T_delay, N_sel = delay_activity_sel.shape
    if T_delay == 0: return 0.0, 0.0, 0.0, 0.0

    # 1. Entropy (E)
    peak_times = np.argmax(delay_activity_sel, axis=0)
    hist_counts, _ = np.histogram(peak_times, bins=np.arange(T_delay + 1))
    prob = hist_counts / np.sum(hist_counts)
    H = entropy(prob, base=2)
    E = H / np.log2(T_delay) if T_delay > 1 else 0

    # 2. Ridge-to-Background (R)
    # Background: 整个试验的平均
    background = np.mean(full_activity_sel, axis=0) + 1e-9
    # Ridge: 峰值附近的平均 (简化为峰值点，更严谨可用 window)
    ridge = np.max(delay_activity_sel, axis=0) 
    R_vals = ridge / background
    R = np.mean(R_vals) 
    # 注意：R值可能很大，论文中进行了归一化，这里为了简便直接用原始定义，或者需要log压缩
    # 论文中 TI = E + R_norm + P_norm。这里我们简化计算，仅做演示。
    # 为了防止 R 过大主导 TI，我们可以取 log10(R) 或者假设 R 已经在合理范围
    R = np.log1p(R) # Log压缩以保持数值稳定

    # 3. Proportion (P)
    # 多少比例的选择性神经元的峰值确实在 Delay 期间？
    # 由于我们是先根据Delay差异筛选的，这个比例通常很高。
    # 这里计算：峰值在Delay期的神经元 / 总神经元N (不仅仅是选择性的)
    total_N = neural_activity_trial.shape[1]
    peak_indices_global = np.argmax(neural_activity_trial, axis=0)
    # 判断峰值是否在 delay 范围内
    is_in_delay = (peak_indices_global >= delay_start) & (peak_indices_global < delay_end)
    # 且属于选择性神经元
    is_selective = np.zeros(total_N, dtype=bool)
    is_selective[selective_indices] = True
    
    count_valid = np.sum(is_in_delay & is_selective)
    P = count_valid / total_N

    TI = E + R + P
    return TI, E, R, P

def calculate_ti_for_trials(all_neural_activities, trial_infos):
    # 1. 筛选神经元
    sel_indices = identify_selective_neurons(all_neural_activities, trial_infos)
    print(f"Selected {len(sel_indices)} memory-selective neurons out of {all_neural_activities[0].shape[1]}")
    
    if len(sel_indices) < 5:
        print("Warning: Too few selective neurons found. TI might be unreliable.")
        return 0.0, []

    ti_scores = []
    for act, info in zip(all_neural_activities, trial_infos):
        start = info['fix_steps'] + info['stim_steps']
        end = start + info['delay_steps']
        ti, _, _, _ = calculate_ti_single_trial(act, sel_indices, start, end)
        ti_scores.append(ti)

    return np.mean(ti_scores), ti_scores

# 保存和绘图函数保持不变，但记得在 main 中调用新的函数
def save_checkpoint(model, avg_ti, W_out, filename_suffix=""):
    """
    保存模型检查点。根据 TI 值自动归类到 high_ti 或 low_ti 文件夹。
    
    Args:
        model: 训练好的 HierarchicalModularReservoir 实例。
        avg_ti (float): 计算出的平均瞬态指数。
        W_out (np.ndarray): 训练好的读出层权重 (N, 3)。
        filename_suffix (str): 文件名后缀，通常包含参数信息。
    """
    # 设定阈值分类 (参考论文：TI > 1.5 为高，TI < 1.0 偏低/持续性)
    if avg_ti >= 1.5:
        save_dir = TI_HIGH_DIR
        prefix = "high_ti"
    elif avg_ti <= 1.0:
        save_dir = TI_LOW_DIR
        prefix = "low_ti"
    else:
        # 中间状态也保存到 checkpoint 根目录
        save_dir = CHECKPOINT_DIR
        prefix = "mid_ti"

    filename = f"{prefix}_model_ti_{avg_ti:.4f}_{filename_suffix}.pkl"
    filepath = os.path.join(save_dir, filename)

    # 构建保存字典
    checkpoint_data = {
        'config': model.config,           # 保存配置对象
        'weights': {
            'W_res': model.W_res,         # 循环权重
            'W_in': model.W_in,           # 输入权重
            'W_out': W_out                # 读出权重 (线性回归结果)
        },
        'metrics': {
            'avg_ti': avg_ti
        }
    }

    try:
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        print(f"Checkpoint saved successfully: {filepath}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
# def plot_neural_activity_heatmap(neural_activity_trial, trial_info, sel_indices=None, title_suffix="", filename_suffix=""):
#     """仅绘制选择性神经元的 Heatmap"""
#     start = trial_info['fix_steps'] + trial_info['stim_steps']
#     end = start + trial_info['delay_steps']
    
#     if sel_indices is not None and len(sel_indices) > 0:
#         activity_to_plot = neural_activity_trial[:, sel_indices]
#     else:
#         activity_to_plot = neural_activity_trial

#     delay_act = activity_to_plot[start:end, :]
#     # 根据峰值时间排序
#     peak_times = np.argmax(delay_act, axis=0)
#     sorted_idx = np.argsort(peak_times)
#     sorted_act = activity_to_plot[:, sorted_idx] # 绘制整个trial，但按delay排序

#     plt.figure(figsize=(10, 6))
#     # 标出 Delay 区域
#     plt.axvline(x=start, color='r', linestyle='--', alpha=0.5)
#     plt.axvline(x=end, color='r', linestyle='--', alpha=0.5)
    
#     plt.imshow(sorted_act.T, aspect='auto', cmap='jet', interpolation='nearest', vmin=0)
#     plt.colorbar(label='Firing Rate (r)')
#     plt.title(f'TRNN Activity (Selective Neurons)\n{title_suffix}')
#     plt.xlabel('Time Steps')
#     plt.ylabel('Neurons (Sorted)')
#     plt.savefig(os.path.join(VISUALIZE_DIR, f"heatmap_{filename_suffix}.png"))
#     plt.close()

def plot_paper_style_heatmap(neural_activity_trial, trial_info, sel_indices=None, normalize=True, filename_suffix=""):
    """
    绘制符合论文 Figure 1e, f 风格的热图。
    
    特点：
    1. 仅包含选择性神经元（如果提供了 sel_indices）。
    2. 神经元根据延迟期内的峰值时间排序。
    3. 绘制表示任务阶段边界的垂直虚线。
    4. 对每个神经元进行归一化，以突出峰值位置。
    
    Args:
        neural_activity_trial (np.ndarray): (Time, Neurons) 原始活动矩阵。
        trial_info (dict): 包含时间步长信息的字典。
        sel_indices (np.ndarray, optional): 选择性神经元的索引列表。
        normalize (bool): 是否将每个神经元的活动归一化到 [0, 1]。推荐 True。
        filename_suffix (str): 文件名后缀。
    """
    # 1. 数据筛选 (Selection)
    if sel_indices is not None and len(sel_indices) > 0:
        # 仅保留选择性神经元
        X = neural_activity_trial[:, sel_indices]
        print(f"Plotting {len(sel_indices)} selective neurons.")
    else:
        # 如果没有提供索引，绘制所有神经元（或前100个以防太拥挤）
        X = neural_activity_trial
        print("Plotting all neurons (no selection provided).")

    T, N = X.shape
    if N == 0:
        print("No neurons to plot.")
        return

    # 2. 计算时间边界 (Boundaries)
    # 任务阶段: Fixation -> Sample -> Delay -> Test -> Response
    t_fix_end = trial_info['fix_steps']
    t_sample_end = t_fix_end + trial_info['stim_steps']
    t_delay_end = t_sample_end + trial_info['delay_steps']
    t_test_end = t_delay_end + trial_info['stim_steps']
    
    # 3. 排序逻辑 (Sorting)
    # 提取延迟期的数据用于计算排序
    delay_activity = X[t_sample_end:t_delay_end, :]
    
    # 找到每个神经元在延迟期内的峰值索引 (argmax)
    # 注意：如果延迟期全是0，argmax会返回0，这没关系
    peak_times_in_delay = np.argmax(delay_activity, axis=0)
    
    # 获取排序索引 (argsort)
    sort_order = np.argsort(peak_times_in_delay)
    
    # 对整个试验的数据进行重排
    X_sorted = X[:, sort_order]

    # 4. 归一化 (Normalization) - 关键步骤
    # 为了让热图好看，每个神经元的活动除以其最大值，使其范围在 [0, 1]
    if normalize:
        max_vals = np.max(X_sorted, axis=0, keepdims=True)
        # 避免除以0
        max_vals[max_vals == 0] = 1.0 
        X_plot = X_sorted / max_vals
    else:
        X_plot = X_sorted

    # 5. 绘图 (Plotting)
    plt.figure(figsize=(10, 8))
    
    # aspect='auto' 保证格子是长方形填满画布
    # origin='lower' 让 y轴索引0在底部，符合常规直觉 (或者根据论文图来看，时间轴是X，神经元是Y)
    # 论文中通常: X轴是时间, Y轴是神经元 #1 到 #N
    # transpose X_plot 因为 imshow 默认第一维是行(Y)，第二维是列(X)
    plt.imshow(X_plot.T, aspect='auto', cmap='jet', interpolation='nearest', origin='lower', vmin=0, vmax=1)
    
    # 绘制垂直虚线 (Boundaries)
    boundaries = [t_fix_end, t_sample_end, t_delay_end, t_test_end]
    labels = ['Sample On', 'Delay Start', 'Test On', 'Resp Start']
    
    for x_pos in boundaries:
        plt.axvline(x=x_pos, color='white', linestyle='--', linewidth=1.5, alpha=0.8)

    # 添加阶段文字标注 (可选)
    # 在时间轴上方标注 Sample, Delay, Test
    y_lim = X_plot.shape[1]
    plt.text(t_fix_end + (t_sample_end-t_fix_end)/2, y_lim * 1.02, 'Sample', ha='center', fontsize=10)
    plt.text(t_sample_end + (t_delay_end-t_sample_end)/2, y_lim * 1.02, 'Delay', ha='center', fontsize=10)
    plt.text(t_delay_end + (t_test_end-t_delay_end)/2, y_lim * 1.02, 'Test', ha='center', fontsize=10)

    # 标题和标签
    info_str = f"Sample: {trial_info['sample_odor']}, Test: {trial_info['test_odor']}, Match: {trial_info['is_match']}"
    plt.title(f"Neural Activity (Sorted by Delay Peak)\n{info_str}", fontsize=12)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Neurons (Sorted)', fontsize=12)
    plt.colorbar(label='Normalized Activity')
    
    # 保存
    save_path = os.path.join(VISUALIZE_DIR, f"paper_style_heatmap_{filename_suffix}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300) # 高分辨率保存
    plt.close()
    print(f"Heatmap saved to: {save_path}")
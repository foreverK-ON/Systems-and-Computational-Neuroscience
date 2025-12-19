import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import entropy, ranksums
from config import VISUALIZE_DIR

def normalize_activity(data):
    """
    仿照 model_torch.py 的 normalization1 和 relu
    data: (Time, Neurons)
    """
    # 1. 减去每个神经元的时间均值
    data_mean = data - np.mean(data, axis=0)
    # 2. 除以最大绝对值 (加上 epsilon 防止除零)
    data_norm = data_mean / (np.max(np.abs(data_mean), axis=0) + 1e-9)
    # 3. ReLU (只保留正向活动)
    return np.maximum(0, data_norm)

def identify_selective_neurons(all_activities, trial_infos, p_value_threshold=0.05):
    """
    筛选记忆选择性神经元 (基于论文逻辑)。
    比较 Sample A 和 Sample B 在延迟期的活动差异。
    """
    delay_activities_A = []
    delay_activities_B = []

    for act, info in zip(all_activities, trial_infos):
        # 提取延迟期平均活动
        delay_start, delay_end = info['indices']['delay']

        mean_delay_act = np.mean(act[delay_start:delay_end, :], axis=0) # (N,)
        
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

    # 1. 提取并预处理数据
    # 注意：参考代码是在整个 Delay 窗口上计算的
    # 我们先切片出 Delay 期的数据，还是对整段处理？
    # model_torch.py line 258: data = relu(normalization1(hidden_act).T)
    # 它是对整个 trace 做归一化。
    
    # 提取选择性神经元
    activity_sel = neural_activity_trial[:, selective_indices] # (T, N_sel)
    
    # 归一化 (关键步骤!)
    activity_norm = normalize_activity(activity_sel)
    
    # 仅关注 Delay 窗口的数据进行指标计算
    # model_torch.py 中 delay 窗口是固定的切片 (20:80)
    delay_data = activity_norm[delay_start:delay_end, :] # (T_delay, N_sel)
    T_delay, N_sel = delay_data.shape
    
    if T_delay < 2: return 0., 0., 0., 0.

    # --- 1. Entropy (E) ---
    # 计算每个神经元在 Delay 期内的峰值时刻
    peak_times = np.argmax(delay_data, axis=0) # (N_sel,)
    
    # 计算直方图
    nbins = min(T_delay, 20) # 保持 bin 数量合理
    hist_counts, _ = np.histogram(peak_times, bins=nbins)
    
    # 计算实际熵
    prob = hist_counts / (np.sum(hist_counts) + 1e-9)
    H_actual = entropy(prob) # 默认 base=e
    
    # 计算最大可能的熵 (均匀分布)
    # model_torch.py: entrpy_max = entropy(np.ones(entrpy_bins)*...)
    # 即 log(nbins)
    H_max = np.log(nbins)
    
    E = H_actual / H_max if H_max > 0 else 0

    # --- 2. Ridge-to-Background (R) ---
    # model_torch.py 使用窗口求和比
    window_size = 4 # 参考代码设定
    r_ratios = []
    
    for i in range(N_sel):
        trace = delay_data[:, i]
        peak_t = peak_times[i]
        
        # 确定窗口范围
        w_start = max(0, peak_t - window_size // 2)
        w_end = min(T_delay, w_start + window_size)
        
        ridge_sum = np.sum(trace[w_start:w_end])
        total_sum = np.sum(trace) + 1e-9
        
        if total_sum == 0:
            r_ratios.append(0)
        else:
            r_ratios.append(ridge_sum / total_sum)
            
    R = np.mean(r_ratios) # 这一步自然就在 [0, 1] 之间

    # --- 3. Proportion (P) ---
    # 检查峰值是否确实在 Delay 期内
    # 在 model_torch.py 中，是看峰值索引是否在规定范围内
    # 我们这里 delay_data 就是 Delay 期的数据，argmax 肯定在范围内？
    # 不完全是。参考代码是看整个 trial 的峰值是否落在 Delay 区间。
    
    full_peak_indices = np.argmax(activity_norm, axis=0) # (N_sel,) 针对整个 Trial
    
    # 判断峰值是否在 [delay_start, delay_end)
    in_delay_mask = (full_peak_indices >= delay_start) & (full_peak_indices < delay_end)
    
    # 分母使用 len(selective_indices) (即 N_sel)
    P = np.sum(in_delay_mask) / N_sel

    # --- Final TI ---
    # 参考代码: Total = SI_trial_vec (R+E) + trans_len (P)
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
        start, end = info['indices']['delay']
        ti, _, _, _ = calculate_ti_single_trial(act, sel_indices, start, end)
        ti_scores.append(ti)

    return np.mean(ti_scores), ti_scores

def plot_paper_style_heatmap(neural_activity_trial, trial_info, sel_indices=None, normalize=True,ti_score = None, filename_suffix=""):
    """
    绘制符合论文 Figure 1e, f 风格的热图。支持自定义 TI 显示和更大的尺寸。
    
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
    indices = trial_info['indices']
    t_fix_end = indices['fix'][1]
    t_sample_end = indices['sample'][1]
    t_delay_end = indices['delay'][1]
    # t_test_end = indices['test'][1]
    
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
    plt.figure(figsize=(14, 10))
    
    # aspect='auto' 保证格子是长方形填满画布
    # origin='lower' 让 y轴索引0在底部，符合常规直觉 (或者根据论文图来看，时间轴是X，神经元是Y)
    # 论文中通常: X轴是时间, Y轴是神经元 #1 到 #N
    # transpose X_plot 因为 imshow 默认第一维是行(Y)，第二维是列(X)
    plt.imshow(X_plot.T, aspect='auto', cmap='jet', interpolation='nearest', origin='lower', vmin=0, vmax=1)
    
    # 绘制垂直虚线 (Boundaries)
    boundaries = [t_fix_end, t_sample_end, t_delay_end]
    
    
    for x_pos in boundaries:
        plt.axvline(x=x_pos, color='white', linestyle='--', linewidth=1.5, alpha=0.8)

    # 添加阶段文字标注 (可选)
    # 在时间轴上方标注 Sample, Delay, Test
    y_lim = X_plot.shape[1]
    plt.text((indices['sample'][0] + indices['sample'][1])/2, y_lim * 1.01, 'Sample', ha='center', fontsize=10)
    plt.text((indices['delay'][0] + indices['delay'][1])/2, y_lim * 1.01, 'Delay', ha='center', fontsize=10)
    plt.text((indices['test'][0] + indices['test'][1])/2, y_lim * 1.01, 'Test', ha='center', fontsize=10)

    # 标题设置 (包含 TI)
    title_lines = []
    # 第一行：TI 指数 (顶部中间，大字体)
    if ti_score is not None:
        title_lines.append(f"Transient Index (TI): {ti_score:.4f}")
    
    # 第二行：试验信息
    info_str = f"Sample: {trial_info['sample_odor']} | Test: {trial_info['test_odor']} | Match: {trial_info['is_match']}"
    title_lines.append(info_str)
    
    plt.title("\n".join(title_lines), fontsize=16, pad=20) # pad增加标题与图的距离

    plt.xlabel('Time Steps', fontsize=14)
    plt.ylabel('Neurons (Sorted by Peak Time)', fontsize=14)
    plt.colorbar(label='Normalized Activity')
    
    save_path = os.path.join(VISUALIZE_DIR, f"heatmap_{filename_suffix}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Heatmap saved to: {save_path}")

def plot_reservoir_connectivity(W_res, filename_suffix=""):
    """
    可视化储备池的连接矩阵。
    修改：不再绘制无意义的稀疏热图，改为绘制稀疏结构图(左)和非零权重分布直方图(右)。
    """
    # 确保 W_res 是 dense array
    if hasattr(W_res, "toarray"):
        W = W_res.toarray()
    else:
        W = W_res

    N = W.shape[0]
    
    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- 图1: 稀疏结构图 (Sparsity Pattern) ---
    # 使用 spy 函数，只绘制非零元素
    # markersize 需要根据神经元数量调整，神经元越多点越小
    ms = 500.0 / N if N > 0 else 1.0
    ax1.spy(W, markersize=ms, color='black')
    
    # 计算稀疏度
    non_zeros = np.count_nonzero(W)
    sparsity = non_zeros / (N * N)
    
    ax1.set_title(f"Connectivity Pattern (Sparsity: {sparsity:.2%})\nBlack dot = Connection", fontsize=14)
    ax1.set_xlabel("Source Neuron")
    ax1.set_ylabel("Target Neuron")

    # --- 图2: 非零权重分布直方图 (Weight Distribution) ---
    # 提取非零权重
    weights = W[W != 0]
    
    if len(weights) > 0:
        # 绘制直方图
        ax2.hist(weights, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        
        # 标注统计信息
        mean_w = np.mean(weights)
        std_w = np.std(weights)
        ax2.set_title(f"Non-zero Weight Distribution\nMean: {mean_w:.4f}, Std: {std_w:.4f}", fontsize=14)
        ax2.set_xlabel("Weight Value")
        ax2.set_ylabel("Count")
        
        # 画一条0线作为参考
        ax2.axvline(0, color='red', linestyle='--', alpha=0.5)
        
        # 只有在权重有正有负时才设置对称范围，否则自适应
        w_max = np.max(np.abs(weights))
        if np.min(weights) < 0 and np.max(weights) > 0:
            ax2.set_xlim(-w_max*1.1, w_max*1.1)
            
    else:
        ax2.text(0.5, 0.5, "Matrix is empty (All zeros)", 
                 ha='center', va='center', fontsize=12)
        ax2.set_title("Weight Distribution")

    # 保存
    save_path = os.path.join(VISUALIZE_DIR, f"connectivity_{filename_suffix}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Connectivity visualization saved to: {save_path}")


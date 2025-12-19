import numpy as np
from config import (
    N_input, VM_KAPPA, VM_AMPLITUDE, INPUT_NOISE
)

def von_mises_tuning(theta, n_neurons, kappa, amplitude):
    """
    生成 Von Mises 调谐曲线产生的群体激活模式。
    
    Args:
        theta (float): 刺激的角度 (0 ~ 2pi).
        n_neurons (int): 输入神经元的数量 (24).
        kappa (float): 浓度参数，控制曲线的宽度.
        amplitude (float): 峰值幅度.
        
    Returns:
        np.ndarray: 形状为 (n_neurons,) 的激活向量.
    """
    # 生成均匀分布的偏好角度 (-pi 到 pi)
    pref_dirs = np.linspace(-np.pi, np.pi, n_neurons, endpoint=False)
    
    # 1. 计算原始 Von Mises 响应
    # cos 的范围是 [-1, 1]，所以 raw_response 的范围是 [exp(-k), exp(k)]
    raw_response = np.exp(kappa * np.cos(theta - pref_dirs))
    
    # 2. 归一化处理
    # 我们希望当 theta == theta_pref 时，值为 1.0 (在乘以 amplitude 之前)
    # 理论最大值是 exp(kappa * 1) = exp(kappa)
    max_val = np.exp(kappa)
    
    # 将峰值缩放到 1.0
    normalized_response = raw_response / max_val
    
    # 3. 应用幅度缩放
    # 最终输出范围: [amplitude * exp(-2k), amplitude]
    response = amplitude * normalized_response

    return response

def generate_odpa_sequence(odors, delay_steps, trial_count, fix_steps=10, stim_steps=10):
    """
    生成符合标准 ODPA/DMS 任务序列。
    修正点：Test 阶段即为 Response 阶段，Fixation 在 Test 开启时关闭。

    Args:
        odors (list): 气味列表，通常为 ['A', 'B'].
        delay_steps (int): 延迟期的时间步长.
        trial_count (int): 生成的试验数量.
        fix_steps (int): 注视期时长.
        stim_steps (int): 样本(Sample)和测试(Test)刺激呈现的时长 (论文通常为 1s -> 10 steps).

    Returns:
        tuple: (input_sequences, target_sequences, trial_info)
    """
    sequences = []
    targets = []
    info = []

    # 定义气味对应的角度 (A: 0, B: pi)
    # 这样它们在圆周上距离最远，区分度最大
    odor_angles = {odors[0]: 0.0, odors[1]: np.pi}

    for i in range(trial_count):
        sample_odor = np.random.choice(odors)
        test_odor = np.random.choice(odors)
        
        # 匹配逻辑：如果样本和测试气味相同，则为 Match
        is_match = (sample_odor == test_odor)

        # 不再有单独的 resp_steps，因为 Test 期间就是响应期
        total_steps = fix_steps + stim_steps + delay_steps + stim_steps 

        # 1. 初始化输入矩阵: (Time, N_input)
        input_seq = np.zeros((total_steps, N_input))
        
        # 2. 添加基线噪声 (模拟神经元自发背景活动或输入噪声)
        input_seq += np.random.normal(0, INPUT_NOISE, (total_steps, N_input))

        # 3. 初始化目标输出: (Time, 3) -> [Fixation, Match, Non-Match]
        target_seq = np.zeros((total_steps, 3))

        # --- 时间段定义 ---
        t_fix_end = fix_steps
        t_sample_end = t_fix_end + stim_steps
        t_delay_end = t_sample_end + delay_steps
        t_test_end = t_delay_end + stim_steps

        # --- 目标输出 (Targets) ---
        # Fixation: 在 Test 之前保持开启 (Fix + Sample + Delay)
        target_seq[0:t_delay_end, 0] = 1.0
        target_seq[t_delay_end:, 0] = 0.0 # Test 阶段 Fixation 熄灭

        # Decision: 仅在 Test 阶段开启
        if is_match:
            target_seq[t_delay_end:t_test_end, 1] = 1.0
        else:
            target_seq[t_delay_end:t_test_end, 2] = 1.0

        # --- 输入刺激 (Inputs) ---
        # Sample Phase
        sample_angle = odor_angles[sample_odor]
        sample_activity = von_mises_tuning(sample_angle, N_input, VM_KAPPA, VM_AMPLITUDE)
        input_seq[t_fix_end:t_sample_end] += sample_activity

        # Test Phase
        test_angle = odor_angles[test_odor]
        test_activity = von_mises_tuning(test_angle, N_input, VM_KAPPA, VM_AMPLITUDE)
        input_seq[t_delay_end:t_test_end] += test_activity

        # 注意：此处移除了 input_seq *= W_in_scale，避免双重缩放

        sequences.append(input_seq)
        targets.append(target_seq)
        info.append({
            'trial_id': i,
            'sample_odor': sample_odor,
            'test_odor': test_odor,
            'is_match': is_match,
            'delay_steps': delay_steps,
            'indices': {
                'fix': (0, t_fix_end),
                'sample': (t_fix_end, t_sample_end),
                'delay': (t_sample_end, t_delay_end),
                'test': (t_delay_end, t_test_end),
                # 兼容旧代码接口，resp 指向 test 阶段
                'resp': (t_delay_end, t_test_end) 
            }
        })

    return sequences, targets, info

def run_model_on_odpa_trials(model, input_sequences, washout_steps):
    """
    在一系列ODPA试验上运行模型，收集储备池状态。

    Args:
        model: 储备池模型实例。
        input_sequences (list of np.ndarray): 输入序列列表。
        washout_steps (int): 每次试验的洗牌步数。

    Returns:
        list of np.ndarray: 每次试验洗牌后的储备池状态列表。
    """
    all_X_states = []
    for inp in input_sequences:
        X_states = model.run(inp, washout=washout_steps)
        all_X_states.append(X_states)
    return all_X_states

def evaluate_accuracy(X_states_list, W_out, trial_infos):
    """
    评估模型在 ODPA 任务上的准确率。
    修正后的评估函数，更稳健的决策逻辑
    Args:
        X_states_list (list): 储备池状态列表.
        W_out (np.ndarray): 训练好的读出权重 (N_hidden, 3).
        trial_infos (list): 试验信息.

    Returns:
        float: 准确率百分比.
    """
    correct_predictions = 0
    total_trials = len(X_states_list)

    for X_states, info in zip(X_states_list, trial_infos):
        # 获取 Test/Response 阶段的时间索引
        resp_start, resp_end = info['indices']['resp']
        
        # 为了避免 Test 开始瞬间的瞬态波动，我们可以取后半段或者整体平均
        # 这里取整体平均
        resp_states = X_states[resp_start:resp_end]
        avg_resp_state = np.mean(resp_states, axis=0) 
        
        # 预测: [Fix, Match, Non-Match]
        logits = avg_resp_state @ W_out 
        
        output_match = logits[1]
        output_nonmatch = logits[2]

        # 简单的 Argmax 逻辑
        pred_is_match = output_match > output_nonmatch
        
        if pred_is_match == info['is_match']:
            correct_predictions += 1
            
    accuracy = (correct_predictions / total_trials) * 100.0
    return accuracy

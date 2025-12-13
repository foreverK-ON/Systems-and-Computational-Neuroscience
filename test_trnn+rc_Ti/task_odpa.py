import numpy as np

def generate_odpa_sequence(odors, delay_steps, trial_count, fix_steps=10, stim_steps=5):
    """
    生成 ODPA 任务序列。

    Args:
        odors (list): 可用的气味列表，例如 ['A', 'B']。
        delay_steps (int): 延迟期的时间步长。
        trial_count (int): 总试验次数。
        fix_steps (int): 固定期时间步长。
        stim_steps (int): 样本和测试刺激时间步长。

    Returns:
        tuple: (input_sequences, target_sequences, trial_info)
    """
    sequences = []
    targets = []
    info = []

    for i in range(trial_count):
        sample_odor = np.random.choice(odors)
        test_odor = np.random.choice(odors)
        is_match = (sample_odor == test_odor)

        resp_steps = 10 # 响应期
        total_steps = fix_steps + stim_steps + delay_steps + stim_steps + resp_steps

        input_seq = np.zeros(total_steps)
        # 目标是3个二进制输出
        target_seq = np.zeros((total_steps, 3)) # [fixation, match, non_match]

        odor_codes = {odors[0]: 1.0, odors[1]: -1.0}

        # 固定期
        target_seq[0:fix_steps, 0] = 1.0
        target_seq[0:fix_steps, 1:] = 0.0

        # 样本期
        sample_start = fix_steps
        sample_end = sample_start + stim_steps
        input_seq[sample_start:sample_end] = odor_codes[sample_odor]
        target_seq[sample_start:sample_end, 0] = 1.0
        target_seq[sample_start:sample_end, 1:] = 0.0

        # 延迟期
        delay_start = sample_end
        delay_end = delay_start + delay_steps
        input_seq[delay_start:delay_end] = 0.0
        target_seq[delay_start:delay_end, 0] = 1.0
        target_seq[delay_start:delay_end, 1:] = 0.0

        # 测试期
        test_start = delay_end
        test_end = test_start + stim_steps
        input_seq[test_start:test_end] = odor_codes[test_odor]
        target_seq[test_start:test_end, 0] = 0.0
        target_seq[test_start:test_end, 1] = 1.0 if is_match else 0.0
        target_seq[test_start:test_end, 2] = 1.0 if not is_match else 0.0

        # 响应期
        resp_start = test_end
        resp_end = resp_start + resp_steps
        input_seq[resp_start:resp_end] = 0.0
        target_seq[resp_start:resp_end, 0] = 0.0
        target_seq[resp_start:resp_end, 1] = 1.0 if is_match else 0.0
        target_seq[resp_start:resp_end, 2] = 1.0 if not is_match else 0.0

        sequences.append(input_seq)
        targets.append(target_seq)
        info.append({
            'trial_id': i,
            'sample_odor': sample_odor,
            'test_odor': test_odor,
            'is_match': is_match,
            'delay_steps': delay_steps,
            'total_steps': total_steps,
            'fix_steps': fix_steps,
            'stim_steps': stim_steps,
            'resp_steps': resp_steps
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

def evaluate_accuracy(X_states_list, W_out_fix, W_out_match, W_out_nonmatch, trial_infos):
    """
    Evaluate the model's accuracy on the ODPA task.

    Args:
        X_states_list (list of np.ndarray): List of reservoir states from trials.
        W_out_fix, W_out_match, W_out_nonmatch (np.ndarray): Readout weights.
        trial_infos (list of dict): Trial information containing true labels.

    Returns:
        float: Accuracy percentage.
    """
    correct_predictions = 0
    total_trials = len(X_states_list)

    for X_states, info in zip(X_states_list, trial_infos):
        # Use the state at the beginning of the test period for prediction
        # Test period starts after fix + stim + delay
        test_start_idx = info['fix_steps'] + info['stim_steps'] + info['delay_steps']
        # Take the state around the middle or slightly after the start of test period
        # Or use the last state of the delay period as input for decision
        # Let's use the first state of the test period for simplicity
        decision_state = X_states[test_start_idx - 1] # State just before test starts, or first of test
        
        # Compute outputs
        out_fix = decision_state @ W_out_fix
        out_match = decision_state @ W_out_match
        out_nonmatch = decision_state @ W_out_nonmatch
        
        # Determine prediction based on highest activation
        # Only match and non_match are relevant for decision
        pred_is_match = out_match > out_nonmatch
        
        # Compare with true label
        true_is_match = info['is_match']
    if pred_is_match == true_is_match:
            correct_predictions += 1
            
    accuracy = (correct_predictions / total_trials) * 100.0
    return accuracy

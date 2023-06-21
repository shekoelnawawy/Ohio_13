import numpy as np


def feature_sum(input_state, indices, dependency_indices):
    a, b, c = dependency_indices
    input_state[indices] = input_state[a] + input_state[b] - input_state[c]
    return input_state


def normalize(input_state, indices, dependency_indices):
    total_sum = np.sum([input_state[i] for i in dependency_indices])

    if total_sum != 0:
        for i in indices:
            input_state[i] = input_state[i] / total_sum

    return input_state


def missing_cgm(input_state, indices, dependency_indices):
    for i in range(len(indices)):
        if input_state[dependency_indices[i]] == 0:
            input_state[indices[i]] = True
        else:
            input_state[indices[i]] = False
    return input_state
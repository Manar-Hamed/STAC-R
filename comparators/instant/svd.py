# Adapted from INSTANT, copyright (c) 2024 Trung-Hieu Tran. MIT; see LICENSE.txt.
# Upstream revision: 8a04bd43c19cc52984a49fb1b7fd1dae14fb2ffc.
import torch
import json
import os
import numpy as np


def SVD_expected_value(input, var=0.95, p=7):
    (U, S, V) = torch.svd(input)
    total_energy = torch.sum(S**2)
    cumulative_energy = torch.cumsum(S**2, dim=0)
    energy_threshold = var * total_energy
    num_singular_values = torch.sum(cumulative_energy <= energy_threshold)
    num_singular_values = num_singular_values + p
    if num_singular_values >= U.shape[1]:
        num_singular_values = U.shape[1]
    if num_singular_values + 1 > 0 and num_singular_values + 1 < U.shape[1]:
        retained_energy = cumulative_energy[num_singular_values + 1]
        U_truncated = U[:, : num_singular_values + 1]
    elif num_singular_values + 1 >= U.shape[1]:
        retained_energy = total_energy
        U_truncated = U[:, :num_singular_values]
    else:
        retained_energy = 0
    retained_energy_per = retained_energy / total_energy
    return U_truncated / torch.sqrt(retained_energy_per)


def log_svd_shapes(iteration, U_dict, output_file):
    if os.path.exists(output_file):
        if os.stat(output_file).st_size == 0:
            all_results = {}
        else:
            with open(output_file, "r") as f:
                all_results = json.load(f)
    else:
        all_results = {}
    if str(iteration) not in all_results:
        all_results[str(iteration)] = {}
    for layer_name, matrix_data in U_dict.items():
        matrix_shape = matrix_data[0].shape
        all_results[str(iteration)][layer_name] = {"shape": matrix_shape}
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4)

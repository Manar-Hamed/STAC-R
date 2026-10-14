# Adapted from INSTANT, copyright (c) 2024 Trung-Hieu Tran. MIT; see LICENSE.txt.
# Upstream revision: 8a04bd43c19cc52984a49fb1b7fd1dae14fb2ffc.
import os
import json
import torch
import random
from transformers import set_seed


def set_train_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_seed(seed)


def log_svd_shapes(iteration, U_dict, output_file):
    all_results = {}
    if os.path.exists(output_file) and os.stat(output_file).st_size > 0:
        try:
            with open(output_file, "r") as f:
                all_results = json.load(f)
        except (json.JSONDecodeError, ValueError):
            all_results = {}
    if str(iteration) not in all_results:
        all_results[str(iteration)] = {}
    for layer_name, matrix_data in U_dict.items():
        matrix_shape = matrix_data.shape
        all_results[str(iteration)][layer_name] = {"shape": matrix_shape}
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4)

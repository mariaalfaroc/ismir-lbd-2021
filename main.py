import os
import random

import numpy as np
import tensorflow as tf

from experimentation import (
    k_fold_experiment,
    k_fold_experiment_scenario_c,
    k_fold_multimodal_experiment,
)
from scenarios.folds_creation import create_a_and_b_folds, create_c_folds

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
tf.config.list_physical_devices("GPU")

# Seed
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

if __name__ == "__main__":
    ##################################### STAND-ALONE EVALUATION:

    EPOCHS = 150
    BATCH_SIZE = {"omr": 16, "amt": 4}
    SYMER_THRESHOLD = 30

    # 1) Evaluate first on Scenario D (original partitions)
    for task in ["omr", "amt"]:
        k_fold_experiment(
            task=task,
            scenario_name="D",
            epochs=EPOCHS,
            batch_size=BATCH_SIZE[task],
        )
    # 2) Create folds for the rest of the scenarios
    create_a_and_b_folds(p_size=2.5, scenario="A")
    create_a_and_b_folds(p_size=4.0, scenario="B")
    create_c_folds(symer_threshold=SYMER_THRESHOLD)
    # 3) Evaluate on the rest of the scenarios
    for scenario in ["A", "B", "C"]:
        for task in ["omr", "amt"]:
            if task == "amt" and scenario in ["A", "B"]:
                continue
            if scenario == "C":
                k_fold_experiment_scenario_c(task=task)
                continue
            k_fold_experiment(
                task=task,
                scenario_name=scenario,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE[task],
            )

    ##################################### MULTIMODAL EVALUATION:

    MATCH = [2, 10, 20, 5]
    MISMATCH = [
        -1,
        5,
        10,
        2,
    ]
    GAP_PENALTY = [-1, -2, -4, -1]

    for s in ["A", "B", "C", "D"]:
        k_fold_multimodal_experiment(
            scenario_name=s, match=MATCH, mismatch=MISMATCH, gap_penalty=GAP_PENALTY
        )

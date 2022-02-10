# -*- coding: utf-8 -*-

import os

import tensorflow as tf

import config
from experimentation import k_fold_experiment, k_fold_experiment_scenario_c

from smith_waterman.prediction_level_fusion import k_fold_multimodal_experiment as sw_k_fold_multimodal_experiment
from scenarios_a_b_c.folds_creation import create_a_and_b_folds, create_c_folds

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
tf.config.list_physical_devices("GPU")

if __name__ == "__main__":
    epochs = 150
    scenarios = ["A", "B", "C", "D"]

    # STAND-ALONE EVALUATION
    # Create folds for Scenarios A and B
    create_a_and_b_folds(p_size=2.5, scenario="A")
    create_a_and_b_folds(p_size=4.0, scenario="B")
    # Evaluate on Scenario D to be able to create Scenario C folds
    config.set_scenario(value="D")
    # OMR
    config.set_task(value="omr")
    config.set_data_globals()
    config.set_arch_globals(batch=16)
    print(f"Task == {config.task}")
    print(f"Scenario == {config.scenario}")
    k_fold_experiment(epochs)
    # AMT
    config.set_task(value="amt")
    config.set_data_globals()
    config.set_arch_globals(batch=4)
    print(f"Task == {config.task}")
    print(f"Scenario == {config.scenario}")
    k_fold_experiment(epochs)
    # Create folds for Scenario C
    create_c_folds()
    # Evaluate on the remaining scenarios
    # NOTE: See notes on k_fold_experiment() and k_fold_experiment_scenario_c() before proceeding!
    for s in scenarios[:-1]:
        config.set_scenario(value=s)
        # OMR
        config.set_task(value="omr")
        config.set_data_globals()
        config.set_arch_globals(batch=16)
        print(f"Task == {config.task}")
        print(f"Scenario == {config.scenario}")
        if s == "C":
            k_fold_experiment_scenario_c()
            # Scenarios A, B, and D are the same for AMT
            config.set_task(value="amt")
            config.set_data_globals()
            config.set_arch_globals(batch=4)
            print(f"Task == {config.task}")
            print(f"Scenario == {config.scenario}")
            k_fold_experiment_scenario_c()
            continue
        k_fold_experiment(epochs)

    # MULTIMODAL EVALUATION
    match = [2, 10, 20, 5]
    mismatch = [-1, 5, 10, 2,]
    gap_penalty = [-1, -2, -4, -1]
    for s in scenarios:
        config.set_scenario(value=s)
        print(f"Scenario{config.scenario}")
        # PREDICTION LEVEL FUSION (SMITH - WATERMAN)
        sw_k_fold_multimodal_experiment(match=match, mismatch=mismatch, gap_penalty=gap_penalty)

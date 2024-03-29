import os
import gc
from typing import List

import pandas as pd
from tensorflow import keras

from networks.models import build_models
from networks.train import train_and_test_model
from networks.test import evaluate_model
from scenarios.folds_creation import (
    get_folds_filenames,
    get_datafold_filenames,
    check_and_retrive_vocabulary,
)
from scenarios.config import MODEL_FROM
from late_fusion.predict import evaluate_multimodal_transcription


######################################################################## STAND-ALONE EVALUATION:


# Utility function for performing a k-fold cross-validation
# experiment on a single dataset (SCENARIOS A, B, and D)
def k_fold_experiment(
    *, task: str, scenario_name: str, epochs: int = 150, batch_size: int = 16
):
    keras.backend.clear_session()
    gc.collect()

    # ---------- PRINT EXPERIMENT DETAILS

    print(f"5-fold cross-validation experiment for scenario {scenario_name}")
    print(f"\tTask: {task}")
    print(f"\tEpochs: {epochs}")
    print(f"\tBatch size: {batch_size}")

    # ---------- FOLDS COLLECTION

    folds = get_folds_filenames(scenario_name)

    # ---------- 5-FOLD EVALUATION

    for id, (train_fold, val_fold, test_fold) in enumerate(
        zip(folds["train"], folds["val"], folds["test"])
    ):
        # With 'clear_session()' called at the beginning,
        # Keras starts with a blank state at each iteration
        # and memory consumption is constant over time
        keras.backend.clear_session()
        gc.collect()

        print(f"Fold {id}")

        # Get the current fold data
        train_images, train_labels = get_datafold_filenames(
            task=task, fold_filename=train_fold
        )
        val_images, val_labels = get_datafold_filenames(
            task=task, fold_filename=val_fold
        )
        test_images, test_labels = get_datafold_filenames(
            task=task, fold_filename=test_fold
        )
        print(f"Train size: {len(train_images)}")
        print(f"Validation size: {len(val_images)}")
        print(f"Test size: {len(test_images)}")

        # Check and retrieve vocabulary
        w2i, i2w = check_and_retrive_vocabulary(fold_id=id)

        # Build the models
        model, prediction_model = build_models(task=task, num_labels=len(w2i))

        # Set filepaths outputs
        output_dir = f"results/scenario{scenario_name}/fold{id}"
        os.makedirs(output_dir, exist_ok=True)
        pred_model_filepath = os.path.join(output_dir, f"best_{task}_model.keras")
        log_path = os.path.join(output_dir, f"{task}_logs.csv")

        # Train, validate, and test models
        # Save logs in CSV file
        train_and_test_model(
            task=task,
            data=(
                train_images,
                train_labels,
                val_images,
                val_labels,
                test_images,
                test_labels,
            ),
            vocabularies=(w2i, i2w),
            epochs=epochs,
            batch_size=batch_size,
            model=model,
            prediction_model=prediction_model,
            pred_model_filepath=pred_model_filepath,
            log_path=log_path,
        )

        # Clear memory
        del train_images, train_labels, val_images, val_labels, test_images, test_labels
        del model, prediction_model


# Utility function for performing a k-fold test partition experiment
# using previously trained models (SCENARIO C)
def k_fold_experiment_scenario_c(*, task: str):
    keras.backend.clear_session()
    gc.collect()

    # ---------- PRINT EXPERIMENT DETAILS

    print("5-fold cross-validation experiment for scenario C")
    print(f"\tTask: {task}")

    # ---------- DATA COLLECTION

    folds = get_folds_filenames("C")

    # ---------- K-FOLD EVALUATION

    for id, test_fold in enumerate(folds["test"]):
        # With 'clear_session()' called at the beginning,
        # Keras starts with a blank state at each iteration
        # and memory consumption is constant over time.
        keras.backend.clear_session()
        gc.collect()

        print(f"Fold {id}")

        # Get the current fold data
        test_images, test_labels = get_datafold_filenames(
            task=task, fold_filename=test_fold
        )
        print(f"Test size: {len(test_images)}")

        # Check and retrieve vocabulary
        i2w = check_and_retrive_vocabulary(fold_id=id)[1]

        # Test the best validation model
        print("Evaluating best validation model over test data")
        pred_model_filepath = f"results/scenario{MODEL_FROM[task]['C']}/fold{id}"
        pred_model_filepath = os.path.join(
            pred_model_filepath, f"best_{task}_model.keras"
        )
        assert os.path.exists(pred_model_filepath), "Model not found!"
        prediction_model = keras.models.load_model(pred_model_filepath)
        test_symer, test_seqer = evaluate_model(
            task=task,
            model=prediction_model,
            images_files=test_images,
            labels_files=test_labels,
            i2w=i2w,
        )

        # Save fold logs
        output_dir = f"results/scenarioC/fold{id}"
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, f"{task}_logs.csv")
        logs = {"test_symer": [test_symer], "test_seqer": [test_seqer]}
        logs = pd.DataFrame.from_dict(logs)
        logs.to_csv(log_path, index=False)

        # Clear memory
        del test_images, test_labels
        del prediction_model


######################################################################## MULTIMODAL EVALUATION:


# Utility function for performing a k-fold cross-validation multimodal experiment on a single dataset
def k_fold_multimodal_experiment(
    *,
    scenario_name: str,
    match: List[int] = [2],
    mismatch: List[int] = [-1],
    gap_penalty: List[int] = [-1],
):
    keras.backend.clear_session()
    gc.collect()

    # ---------- PRINT EXPERIMENT DETAILS

    print(f"5-fold multimodal cross-validation experiment for scenario {scenario_name}")
    print(f"\tMatch values: {match}")
    print(f"\tMismatch values: {mismatch}")
    print(f"\tGap penalty values: {gap_penalty}")

    # ---------- DATA COLLECTION

    folds = get_folds_filenames(scenario_name)

    # ---------- K-FOLD EVALUATION

    for id, test_fold in enumerate(folds["test"]):
        # With 'clear_session()' called at the beginning,
        # Keras starts with a blank state at each iteration
        # and memory consumption is constant over time.
        keras.backend.clear_session()
        gc.collect()

        print(f"Fold {id}")

        # Get the current fold data
        omr_test_images, test_labels = get_datafold_filenames(
            task="omr", fold_filename=test_fold
        )
        amt_test_images, _ = get_datafold_filenames(task="amt", fold_filename=test_fold)
        assert len(omr_test_images) == len(
            amt_test_images
        ), "Different number of files in OMR and AMT test sets!"
        print(f"Test size: {len(omr_test_images)}")

        # Check and retrieve vocabulary
        i2w = check_and_retrive_vocabulary(fold_id=id)[1]

        # Load the models
        omr_pred_model_filepath = os.path.join(
            "results", f"scenario{MODEL_FROM['omr'][scenario_name]}"
        )
        omr_pred_model_filepath = os.path.join(
            omr_pred_model_filepath, f"fold{id}", "best_omr_model.keras"
        )
        omr_prediction_model = keras.models.load_model(omr_pred_model_filepath)

        amt_pred_model_filepath = os.path.join(
            "results", f"scenario{MODEL_FROM['amt'][scenario_name]}"
        )
        amt_pred_model_filepath = os.path.join(
            amt_pred_model_filepath, f"fold{id}", "best_amt_model.keras"
        )
        amt_prediction_model = keras.models.load_model(amt_pred_model_filepath)

        # Set filepaths outputs
        output_dir = f"results/scenario{scenario_name}/fold{id}"
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, "multimodal_logs.csv")

        symer_acc = []
        seqer_acc = []
        # Iterate over the different match, mismatch and gap_penalty values
        for m, mism, gp in zip(match, mismatch, gap_penalty):
            # Multimodal transcription evaluation
            symer, seqer = evaluate_multimodal_transcription(
                omr_model=omr_prediction_model,
                amt_model=amt_prediction_model,
                omr_image_files=omr_test_images,
                amt_image_files=amt_test_images,
                labels_files=test_labels,
                i2w=i2w,
                match=m,
                mismatch=mism,
                gap_penalty=gp,
            )
            symer_acc.append(symer)
            seqer_acc.append(seqer)
        # Save fold logs
        logs = {
            "match": match,
            "mismatch": mismatch,
            "gap_penalty": gap_penalty,
            "symer": symer_acc,
            "seqer": seqer_acc,
        }
        logs = pd.DataFrame.from_dict(logs)
        logs.to_csv(log_path, index=False)

        # Clear memory
        del omr_test_images, amt_test_images, test_labels
        del omr_prediction_model, amt_prediction_model

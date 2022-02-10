# -*- coding: utf-8 -*-

import os, gc, shutil

import pandas as pd
import numpy as np
from tensorflow import keras

import config
from data_processing import get_folds_filenames, get_datafolds_filenames, get_fold_vocabularies, save_w2i_dictionary, load_dictionaries, train_data_generator
from models import build_models
from evaluation import evaluate_model

# Utility function for training, validating, and testing a model and saving the logs in a CSV file
def train_and_test_model(data, vocabularies, epochs, model, prediction_model, pred_model_filepath, log_path):
    train_images, train_labels, val_images, val_labels, test_images, test_labels = data
    w2i, i2w = vocabularies

    # Instantiate logs variables
    loss_acc = []
    val_symer_acc = []
    val_seqer_acc = []

    # Train and validate
    best_symer = np.Inf
    best_epoch = 0
    for epoch in range(epochs):
        print(f"--Epoch {epoch + 1}--")
        print("Training:")
        history = model.fit(
            train_data_generator(train_images, train_labels, w2i),
            epochs=1, 
            verbose=2,
            steps_per_epoch=len(train_images) // config.batch_size
        )
        loss_acc.extend(history.history["loss"])
        print("Validating:")
        val_symer, val_seqer = evaluate_model(prediction_model, val_images, val_labels, i2w)
        val_symer_acc.append(val_symer)
        val_seqer_acc.append(val_seqer)
        if val_symer < best_symer:
            best_symer = val_symer
            best_epoch = epoch
            print(f"Saving new best prediction model to file {pred_model_filepath}")
            prediction_model.save(filepath=pred_model_filepath)
    print(f"Best validation SymER (%): {best_symer:.2f} at epoch {best_epoch + 1}")

    # Test the best validation model
    print("Evaluating best validation model over test data")
    prediction_model = keras.models.load_model(pred_model_filepath)
    test_symer, test_seqer = evaluate_model(prediction_model, test_images, test_labels, i2w)

    # Save fold logs
    # The last line on the CSV file is the one corresponding to the best validation model
    loss_acc.extend(["-", loss_acc[best_epoch]])
    val_symer_acc.extend(["-", val_symer_acc[best_epoch]])
    val_seqer_acc.extend(["-", val_seqer_acc[best_epoch]])
    logs = {
        "loss" : loss_acc, 
        "val_symer": val_symer_acc, "val_seqer": val_seqer_acc, 
        "test_symer": ["-"] * (len(val_symer_acc) - 1) + [test_symer], "test_seqer": ["-"] * (len(val_seqer_acc) - 1) + [test_seqer]
        }
    logs = pd.DataFrame.from_dict(logs)
    logs.to_csv(log_path, index=False)

    return

# -- EXPERIMENT TYPES -- #

# Utility function for performing a k-fold cross-validation experiment on a single dataset (SCENARIOS A, B, and D)
# NOTE: We MUST first evaluate both tasks on Scenario D and then manually copy the files for AMT to Scenarios A and B
def k_fold_experiment(epochs):
    keras.backend.clear_session()
    gc.collect()

    # ---------- PRINT EXPERIMENT DETAILS

    print("k-fold cross-validation experiment")
    print(f"Data used {config.base_dir.stem}")

    # ---------- DATA COLLECTION

    train_folds_files = get_folds_filenames("train")
    val_folds_files = get_folds_filenames("val")
    test_folds_files = get_folds_filenames("test")

    assert len(train_folds_files) == len(val_folds_files) == len(test_folds_files)

    train_images_fnames, train_labels_fnames = get_datafolds_filenames(train_folds_files)
    val_images_fnames, val_labels_fnames = get_datafolds_filenames(val_folds_files)
    test_images_fnames, test_labels_fnames = get_datafolds_filenames(test_folds_files) 

    # ---------- K-FOLD EVALUATION

    # Start the k-fold evaluation scheme
    # k = len(train_images_fnames)
    # for i in range(k):
    for i in range(1):
        # With 'clear_session()' called at the beginning,
        # Keras starts with a blank state at each iteration
        # and memory consumption is constant over time.
        keras.backend.clear_session()
        gc.collect()

        print(f"Fold {i}")

        # Set filepaths outputs
        output_dir = config.output_dir / config.task / f"Fold{i}"
        os.makedirs(output_dir, exist_ok=True)
        pred_model_filepath = output_dir / "best_model.keras"
        w2i_filepath = output_dir / "w2i.json"
        log_path = output_dir / "logs.csv"

        # Get the current fold data
        train_images, train_labels = train_images_fnames[i], train_labels_fnames[i]
        val_images, val_labels = val_images_fnames[i], val_labels_fnames[i]
        test_images, test_labels = test_images_fnames[i], test_labels_fnames[i]

        assert len(train_images) == len(train_labels)
        assert len(val_images) == len(val_labels)
        assert len(test_images) == len(test_labels)

        print(f"Train: {len(train_images)}")
        print(f"Validation: {len(val_images)}")
        print(f"Test: {len(test_images)}")

        if config.scenario == "D":
            # Get and save vocabularies
            w2i, i2w = get_fold_vocabularies(train_labels)
            save_w2i_dictionary(w2i, w2i_filepath)
        else:
            # Load dictionary from Scenario D
            # To use the same vocabulary across scenarios and tasks
            w2i_filepath_ScenarioD = str(w2i_filepath).replace(f"Scenario{config.scenario}", "ScenarioD")
            print(f"Using vocabulary from {w2i_filepath_ScenarioD}")
            w2i, i2w = load_dictionaries(w2i_filepath_ScenarioD)
            shutil.copy(w2i_filepath_ScenarioD, w2i_filepath)

        # Build the models
        model, prediction_model = build_models(num_labels=len(w2i))

        # Train, validate, and test models
        # Save logs in CSV file
        train_and_test_model(
            data=(train_images, train_labels, val_images, val_labels, test_images, test_labels),
            vocabularies=(w2i, i2w),
            epochs=epochs,
            model=model, prediction_model=prediction_model,
            pred_model_filepath=pred_model_filepath, 
            log_path=log_path
        )

        # Clear memory
        del train_images, train_labels, val_images, val_labels, test_images, test_labels
        del model, prediction_model

    return

# Utility function for performing a k-fold test partition experiment using previously trained models (SCENARIO C)
# NOTE: We MUST first evaluate both tasks on Scenario B and then manually copy the files to Scenario C
def k_fold_experiment_scenario_c():
    keras.backend.clear_session()
    gc.collect()

    # ---------- PRINT EXPERIMENT DETAILS

    assert config.scenario == "C"
    print("k-fold test performance experiment")
    print(f"Data used {config.base_dir.stem}")

    # ---------- DATA COLLECTION

    test_folds_files = get_folds_filenames("test")
    test_images_fnames, test_labels_fnames = get_datafolds_filenames(test_folds_files) 

    # ---------- K-FOLD EVALUATION

    # Start the k-fold evaluation scheme
    # k = len(test_images_fnames)
    # for i in range(k):
    for i in range(1):
        # With 'clear_session()' called at the beginning,
        # Keras starts with a blank state at each iteration
        # and memory consumption is constant over time.
        keras.backend.clear_session()
        gc.collect()

        print(f"Fold {i}")

        # Set filepaths outputs
        output_dir = config.output_dir / config.task / f"Fold{i}"
        pred_model_filepath = output_dir / "best_model.keras"
        w2i_filepath = output_dir / "w2i.json"
        log_path = output_dir / "logs.csv"

        # Get the current fold data
        test_images, test_labels = test_images_fnames[i], test_labels_fnames[i]
        assert len(test_images) == len(test_labels)
        print(f"Test: {len(test_images)}")

        # Load dictionary
        i2w = load_dictionaries(w2i_filepath)[1]

        # Test the best validation model
        print("Evaluating best validation model over test data")
        prediction_model = keras.models.load_model(pred_model_filepath)
        test_symer, test_seqer = evaluate_model(prediction_model, test_images, test_labels, i2w)

        # Save fold logs
        logs = {"test_symer": [test_symer], "test_seqer": [test_seqer]}
        logs = pd.DataFrame.from_dict(logs)
        logs.to_csv(log_path, index=False)

        # Clear memory
        del test_images, test_labels
        del prediction_model

    return

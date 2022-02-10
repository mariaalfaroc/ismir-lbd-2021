# -*- coding: utf-8 -*-

import os, gc, random, shutil, pathlib

import numpy as np
from tensorflow import keras

import config
from data_processing import get_folds_filenames, get_datafolds_filenames, load_dictionaries, preprocess_image, preprocess_label
from evaluation import ctc_greedy_decoder, compute_metrics

# Multimodal image and audio music transcription
# Carlos de la Fuente, Jose J. Valero-Mas, Francisco J. Castellanos, and Jorge Calvo-Zaragoza 

# There are four scenarios:
# A) OMR Sym-Err ~ AMT Sym-Err
# B) OMR Sym-Err < AMT Sym-Err
# C) (OMR Sym-Err ~ AMT Sym-Err) << 1) scenario
# D) OMR Sym-Err << AMT Sym-Err

#                   Scenario A                          Scenario B                      Scenario C                      Scenario D	
#               OMR	                AMT             OMR	            AMT	            OMR	            AMT             OMR         AMT
# Train	    2.5% Part. Orig.    Part. Orig.	    4% Part. Orig.  Part. Orig.     4% Part. Orig.  Part. Orig.	    Part. Orig.	Part. Orig.
# Val	    Part. Orig.	        Part. Orig.	    Part. Orig.	    Part. Orig.	    Part. Orig.	    Part. Orig.	    Part. Orig.	Part. Orig.
# Test	    Part. Orig.	        Part. Orig.	    Part. Orig.	    Part. Orig.	    New set	        New set	        Part. Orig.	Part. Orig.

# New set == Created according to AMT model performace -> Samples of the original test partition whose Symbol Error Rate is lower than or equal to 30% 

# -------------------- SCENARIOS A AND B

# OMR data:
# For each train partition of Scenario D, we randomly select either 2.5% or 4%, as corresponds -> Those are the new train partitions
# Validation and test partitions are those of Sceneario D

# Utility function for creating 5-folds with train, validation, and test partitions for Scenarios A and B
def create_a_and_b_folds(p_size: float, scenario: str):
    # Obtain folds for ScenarioD
    config.set_scenario(value="D")
    train_folds_files = get_folds_filenames("train")
    val_folds_files = get_folds_filenames("val")
    test_folds_files = get_folds_filenames("test")
    # Create Scenario{scenario} folder
    os.makedirs(str(config.folds_dir).replace("ScenarioD", f"Scenario{scenario}"), exist_ok=True)
    # Copy val and test folds
    for val, test in zip(val_folds_files, test_folds_files):
        shutil.copyfile(val, val.replace("ScenarioD", f"Scenario{scenario}"))
        shutil.copyfile(test, test.replace("ScenarioD", f"Scenario{scenario}"))
    # Create new train folds
    for i in train_folds_files:
        data = open(i).readlines()
        random.shuffle(data)
        new_size = int(len(data) * p_size / 100)
        data = data[:new_size]
        data = data[:-1] + [data[-1].split("\n")[0]]
        with open(i.replace("ScenarioD", f"Scenario{scenario}"), "w") as txt:
            for s in data:
                txt.write(s)
    return

# -------------------- SCENARIO C

# Utility function for writing 5-folds with train, validation, and test partitions for Scenario C
def write_c_folds(samples: list):
    # Obtain folds for ScenarioB
    config.set_scenario(value="B")
    train_folds_files = get_folds_filenames("train")
    val_folds_files = get_folds_filenames("val")
    # Create ScenarioC folder
    os.makedirs(str(config.folds_dir).replace("ScenarioB", "ScenarioC"), exist_ok=True)
    # Copy train and val folds
    for train, val in zip(train_folds_files, val_folds_files):
        shutil.copyfile(train, train.replace("ScenarioB", "ScenarioC"))
        shutil.copyfile(val, val.replace("ScenarioB", "ScenarioC"))
    # Create new test folds
    config.set_scenario(value="C")
    for id, test in enumerate(samples):
        test_fold = os.path.join(config.folds_dir, f"test_gt_fold{id}.dat")
        # Write folds files
        with open(test_fold, "w") as txt:
            test = [s + "\n" for s in test[:-1]] + [test[-1]]
            txt.writelines(test)
    return

# Utility function for evaluating a model over a dataset and adding the samples that are lower or equal than a threshold to a list
def evaluate_model(model, images_files, labels_files, i2w, symer_threshold=30):
    new_set = []
    # Iterate over images
    for i in range(len(images_files)):
        images, images_len = list(zip(*[preprocess_image(images_files[i])]))
        images = np.array(images, dtype="float32")
        # Obtain predictions
        y_pred = model(images, training=False)
        # CTC greedy decoder (merge repeated, remove blanks, and i2w conversion)
        y_pred = ctc_greedy_decoder(y_pred, images_len, i2w)
        # Obtain true labels
        y_true = [preprocess_label(labels_files[i], training=False, w2i=None)]
        # Compute Symbol Error Rate
        symer = compute_metrics(y_true, y_pred)[0]
        # If the Symbol Error Rate is lower than or equal to the threshold, the sample gets added to the new subset
        if symer <= symer_threshold:
            new_set.append(pathlib.Path(labels_files[i]).stem)
    print(f"For this fold, only {len(new_set)} samples have a Symbol Error Rate lower than or equal to {symer_threshold}")
    return new_set

# Utility function for obtaining model predictions and creating a new subset based on the corresponding error prediction
def create_c_folds(symer_threshold=30):
    keras.backend.clear_session()
    gc.collect()

    # ---------- PRINT EXPERIMENT DETAILS

    print("k-fold test performance experiment")
    print(f"Data used {config.base_dir.stem}")

    # ---------- DATA COLLECTION

    test_folds_files = get_folds_filenames("test")
    test_images_fnames, test_labels_fnames = get_datafolds_filenames(test_folds_files) 

    # ---------- K-FOLD EVALUATION
    new_set = []

    # Start the k-fold evaluation scheme
    k = len(test_images_fnames)
    for i in range(k):
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

        # Get the current fold data
        test_images, test_labels = test_images_fnames[i], test_labels_fnames[i]
        assert len(test_images) == len(test_labels)
        print(f"Test: {len(test_images)}")

        # Get and save vocabularies
        i2w = load_dictionaries(w2i_filepath)[1]

        # Get prediction model
        prediction_model = keras.models.load_model(pred_model_filepath)

        # Evaluate model and add to the new set the samples whose prediction error is lower than or equal to threshold
        new_set.append(evaluate_model(prediction_model, test_images, test_labels, i2w, symer_threshold=symer_threshold))

        # Clear memory
        del test_images, test_labels
        del prediction_model

    # Create 5-folds using new set samples
    write_c_folds(samples=new_set)
        
    return

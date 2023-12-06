import os
import gc
import shutil
import random
from typing import List, Dict, Tuple

from tensorflow import keras

from networks.test import evaluate_model
from my_utils.vocabulary import check_and_retrive_vocabulary

INPUT_EXTENSION = {"omr": "_distorted.jpg", "amt": ".wav"}
LABEL_EXTENSION = ".semantic"


###########################################################################################

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
###########################################################################################


# Get all the folds filenames for each data partition
# folds = {"train": [".../train_gt_fold0.dat", ".../train_gt_fold1.dat", ...], "val": [...], "test": [...]}
def get_folds_filenames(scenario_name: str) -> Dict[str, List[str]]:
    scenario_dir = f"Scenario{scenario_name}"

    folds = {"train": [], "val": [], "test": []}
    for fname in os.listdir(scenario_dir):
        if fname.startswith("train"):
            folds["train"].append(os.path.join(scenario_dir, fname))
        elif fname.startswith("val"):
            folds["val"].append(os.path.join(scenario_dir, fname))
        elif fname.startswith("test"):
            folds["test"].append(os.path.join(scenario_dir, fname))

    assert (
        len(folds["train"]) == len(folds["val"]) == len(folds["test"])
    ), "Folds are not balanced!"

    return {k: sorted(v) for k, v in folds.items()}


# Get all images and labels filenames
# of a corresponding fold filename
def get_datafold_filenames(
    task: str, fold_filename: list
) -> Tuple[List[str], List[str]]:
    images_filenames = []
    labels_filenames = []
    with open(fold_filename) as f:
        lines = f.read().splitlines()
    for line in lines:
        common_path = f"dataset/Corpus/{line}/{line}"
        images_filenames.append(common_path + INPUT_EXTENSION[task])
        labels_filenames.append(common_path + LABEL_EXTENSION)
    return images_filenames, labels_filenames


##################################### SCENARIOS A AND B:

# OMR data:
# For each train partition of Scenario D,
# we randomly select either 2.5% or 4%, as corresponds
# Those are the new train partitions
# Validation and test partitions are those of Sceneario D


# Utility function for creating 5-folds with train,
# validation, and test partitions for Scenarios A and B
def create_a_and_b_folds(p_size: float, scenario: str):
    # Obtain folds for ScenarioD
    folds = get_folds_filenames("D")
    # Create Scenario{scenario} folder
    os.makedirs(f"Scenario{scenario}", exist_ok=True)
    # Copy val and test folds
    for f in folds["val"] + folds["test"]:
        shutil.copyfile(f, f.replace("ScenarioD", f"Scenario{scenario}"))
    # Create new train folds
    for f in folds["train"]:
        with open(f, "r") as dat:
            samples = dat.read().splitlines()
        new_size = int(len(samples) * p_size / 100)
        new_samples = random.sample(samples, k=new_size)
        with open(f.replace("ScenarioD", f"Scenario{scenario}"), "w") as new_dat:
            new_dat.write("\n".join(new_samples))


##################################### SCENARIO C:


# Utility function for writing 5-folds with train,
# validation, and test partitions for Scenario C
def write_c_folds(test_samples: dict):
    # Obtain folds for ScenarioB
    folds = get_folds_filenames("B")
    # Create ScenarioC folder
    os.makedirs("ScenarioC", exist_ok=True)
    # Copy train and val folds
    for f in folds["train"] + folds["val"]:
        shutil.copyfile(f, f.replace("ScenarioB", "ScenarioC"))
    # Create new test folds
    for id, samples in test_samples.items():
        with open(f"ScenarioC/test_gt_fold{id}.dat", "w") as dat:
            dat.write("\n".join(samples))


# Utility function for evaluating a model over a dataset and
# adding the samples that are lower or equal than a threshold to a list
def filter_samples(
    *,
    task: str,
    model: keras.Model,
    images_files: List[str],
    labels_files: List[str],
    i2w: Dict[int, str],
    symer_threshold: int = 30,
):
    new_set = []
    # Iterate over images
    for img, label in zip(images_files, labels_files):
        symer, _ = evaluate_model(
            task=task, model=model, images_files=[img], labels_files=[label], i2w=i2w
        )
        # If the Symbol Error Rate is lower than or equal to the threshold,
        # the sample gets added to the new subset
        if symer <= symer_threshold:
            new_set.append(os.path.splitext(os.path.basename(label))[0])
    print(
        f"For this fold, only {len(new_set)} samples have a Symbol Error Rate lower than or equal to {symer_threshold}"
    )
    return new_set


# Utility function for obtaining model predictions and
# creating a new subset based on the corresponding error prediction
def create_c_folds(symer_threshold: int = 30):
    keras.backend.clear_session()
    gc.collect()

    task = "amt"

    # ---------- PRINT EXPERIMENT DETAILS

    print("5-fold test performance experiment to create scenario C")
    print(f"\tTask: {task}")
    print(f"\tSym-ER threshold: {symer_threshold}")

    # ---------- FOLDS COLLECTION

    folds = get_folds_filenames("D")  # Original test partition

    # ---------- 5-FOLD EVALUATION

    new_set = {}
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
        _, i2w = check_and_retrive_vocabulary(fold_id=id)

        # Get prediction model
        pred_model_filepath = f"results/scenarioD/fold{id}"
        pred_model_filepath = os.path.join(
            pred_model_filepath, f"best_{task}_model.keras"
        )
        assert os.path.exists(pred_model_filepath), "Model from scenario D not found!"
        prediction_model = keras.models.load_model(pred_model_filepath)

        # Evaluate model and add to the new set the samples whose prediction error is lower than or equal to threshold
        new_set[id] = filter_samples(
            task,
            prediction_model,
            test_images,
            test_labels,
            i2w,
            symer_threshold,
        )

        # Clear memory
        del test_images, test_labels
        del prediction_model

    # Create 5-folds using new set samples
    write_c_folds(new_set)

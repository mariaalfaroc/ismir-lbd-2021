import os
import json
from typing import List, Dict, Tuple

from scenarios.folds_creation import get_folds_filenames, get_datafold_filenames

VOCABS_DIR = "./scenarios/vocabs/"
os.makedirs(VOCABS_DIR, exist_ok=True)


def check_and_retrive_vocabulary(fold_id: int) -> Tuple[Dict[str, int], Dict[int, str]]:
    w2i_path = os.path.join(VOCABS_DIR, f"w2i_fold{fold_id}.json")
    if os.path.exists(w2i_path):
        w2i, i2w = load_dictionaries(filepath=w2i_path)
    else:
        # Use ScenarioD train files
        folds = get_folds_filenames("D")
        _, labels_filenames = get_datafold_filenames(
            task="omr", fold_filename=folds["train"][fold_id]
        )
        w2i, i2w = get_fold_vocabularies(labels_filenames)
        save_w2i_dictionary(w2i, filepath=w2i_path)
    return w2i, i2w


# Get dictionaries for w2i and i2w conversion
# corresponding to a single training fold
def get_fold_vocabularies(
    train_labels_fnames: List[str],
) -> Tuple[Dict[str, int], Dict[int, str]]:
    # Get all tokens related to a SINGLE train data fold
    tokens = []
    for fname in train_labels_fnames:
        with open(fname) as f:
            tokens.extend(f.read().split())
    # Eliminate duplicates and sort them alphabetically
    tokens = sorted(set(tokens))
    # Create vocabularies
    w2i = dict(zip(tokens, range(len(tokens))))
    i2w = dict(zip(range(len(tokens)), tokens))
    return w2i, i2w


# Utility function for saving w2i dictionary in a JSON file
def save_w2i_dictionary(w2i: Dict[str, int], filepath: str):
    # Save w2i dictionary to JSON filepath to later retrieve it
    # No need to save both of them as they are related
    with open(filepath, "w") as json_file:
        json.dump(w2i, json_file)


# Retrieve w2i and i2w dictionaries from w2i JSON file
def load_dictionaries(filepath: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(filepath, "r") as json_file:
        w2i = json.load(json_file)
    i2w = {int(v): k for k, v in w2i.items()}
    return w2i, i2w

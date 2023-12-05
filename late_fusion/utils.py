from typing import List, Dict, Tuple

from tensorflow import keras
import numpy as np

from my_utils.preprocessing import preprocess_input


# Utility function for merging consecutive repeated values in a sequence
# and averaging their correspoding probabilities
def merge_repeated(
    y_dec: np.ndarray, y_prob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    prev_value = None
    prob_acc = 0
    repetitions = 0
    y_dec_uniq = []
    y_prob_uniq = []
    for y, prob in zip(y_dec, y_prob):
        # Same value as before
        if y == prev_value:
            # Delete previous appended probability
            del y_prob_uniq[-1]
            prob_acc += prob
            repetitions += 1
            # The new probability associated to the value is the average of the probabilities
            y_prob_uniq.append(prob_acc / repetitions)
        # Different value from before
        else:
            # Append the value
            y_dec_uniq.append(y)
            # Append its correspoding probability
            y_prob_uniq.append(prob)
            prob_acc = prob
            repetitions = 1
            prev_value = y
    assert len(y_dec_uniq) == len(y_prob_uniq)
    return np.array(y_dec_uniq), np.array(y_prob_uniq)


# Utility function for filtering out a value and its associated probability from a pair of arrays
def filter_out_value(
    y_dec: np.ndarray, y_prob: np.ndarray, blank_index: int
) -> Tuple[np.ndarray, np.ndarray]:
    not_blank_ids = np.where(y_dec != blank_index)
    return y_dec[not_blank_ids], y_prob[not_blank_ids]


# CTC-greedy decoder (merge repeated elements, remove blank labels, and covert back to string labels)
# Returns such best path and its probability array
def manual_ctc_greedy_decoder(
    y_pred: np.ndarray, input_length: List[int], i2w: Dict[int, str]
) -> Tuple[List[List[str]], List[List[float]]]:
    # Obtain the best path (class of highest probability at each time step) for each batch item
    y_pred_best = np.argmax(y_pred, axis=-1)
    # Obtain the associated probability for each time step of the best path of each batch item
    y_pred_best_prob = np.amax(y_pred, axis=-1)
    # Merge consecutive repeated classes and filter out blank label class
    y_pred_dec = []
    y_pred_prob_dec = []
    for y, prob, length in zip(y_pred_best, y_pred_best_prob, input_length):
        y, prob = merge_repeated(y[:length], prob[:length])
        y, prob = filter_out_value(y, prob, blank_index=len(i2w))
        # Append to accumulator variables
        y_pred_dec.append([i2w[i] for i in y])
        y_pred_prob_dec.append(prob.tolist())
    return y_pred_dec, y_pred_prob_dec


# Utility function for obtaining the predictions and their associated probability arrays of a dataset
def get_predictions_and_probabilities(
    *, task: str, model: keras.Model, images_files: List[str], i2w: Dict[int, str]
) -> Tuple[List[List[str]], List[List[float]]]:
    y_pred_acc = []
    y_pred_prob_acc = []
    # Iterate over images
    for img in images_files:
        images, images_len = preprocess_input(task=task, image_path=img)
        images = np.expand_dims(images, axis=0)
        # Obtain predictions
        y_pred = model(images, training=False)
        # Manual CTC greedy decoder
        y_pred, y_pred_prob = manual_ctc_greedy_decoder(
            y_pred.numpy(), [images_len], i2w
        )
        y_pred_acc.extend(y_pred)
        y_pred_prob_acc.extend(y_pred_prob)
    return y_pred_acc, y_pred_prob_acc

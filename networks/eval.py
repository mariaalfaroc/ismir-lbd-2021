from typing import List, Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from networks.metrics import compute_metrics
from my_utils.preprocessing import preprocess_image, preprocess_label


# CTC-greedy decoder:
# 1) Merge repeated elements
# 2) Remove blank labels
# 3) Convert back to string labels
def ctc_greedy_decoder(
    y_pred: tf.Tensor, input_length: List[int], i2w: Dict[int, str]
) -> List[List[str]]:
    input_length = tf.constant(input_length, dtype="int32", shape=(len(input_length),))
    # Blank labels are returned as -1
    y_pred = keras.backend.ctc_decode(y_pred, input_length, greedy=True)[0][0].numpy()
    # i2w conversion
    y_pred = [[i2w[int(i)] for i in b if int(i) != -1] for b in y_pred]
    return y_pred


# Utility function for evaluting a model over a dataset and computing the corresponding metrics
def evaluate_model(
    task: str,
    model: keras.Model,
    images_files: List[str],
    labels_files: List[str],
    i2w: Dict[int, str],
    batch_size: int,
) -> Tuple[float, float]:
    y_pred_acc = []
    # Iterate over images in batches
    for start in range(0, len(images_files), batch_size):
        images, images_len = list(
            zip(
                *[
                    preprocess_image(task=task, image_path=i)
                    for i in images_files[start : start + batch_size]
                ]
            )
        )
        # Zero-pad images to maximum batch image width
        max_width = max(images, key=np.shape).shape[1]
        images = np.array(
            [
                np.pad(i, pad_width=((0, 0), (0, max_width - i.shape[1]), (0, 0)))
                for i in images
            ],
            dtype="float32",
        )
        # Obtain predictions
        y_pred = model(images, training=False)
        # CTC greedy decoder (merge repeated, remove blanks, and i2w conversion)
        y_pred_acc.extend(ctc_greedy_decoder(y_pred, images_len, i2w))
    # Obtain true labels
    y_true_acc = [
        preprocess_label(label_path=i, training=False, w2i=None) for i in labels_files
    ]
    # Compute metrics
    symer, seqer = compute_metrics(y_true_acc, y_pred_acc)
    print(
        f"SymER (%): {symer:.2f}, SeqER (%): {seqer:.2f} - From {len(y_true_acc)} samples"
    )
    return symer, seqer

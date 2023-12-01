from typing import List, Dict, Tuple, Union

import cv2
import numpy as np
from sklearn.utils import shuffle

from networks.models import INPUT_HEIGHT, POOLING_FACTORS


IMAGE_FLAGS = {"omr": 1, "amt": -1}  # 1 -> cv2.IMREAD_COLOR; -1 -> cv2.IMREAD_UNCHANGED


################################################################# INPUT DATA PREPROCESSING:


# Preprocess image:
# 1) Read from path
# 2) Convert to grayscale
# 3) Normalize
# 4) Resize preserving aspect ratio
def preprocess_image(task: str, image_path: str) -> Tuple[np.ndarray, int]:
    img = cv2.imread(image_path, IMAGE_FLAGS[task])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = (255.0 - img) / 255.0
    new_height = INPUT_HEIGHT[task]
    new_width = int(new_height * img.shape[1] / img.shape[0])
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    img = img.reshape(new_height, new_width, 1)
    return img, img.shape[1] // POOLING_FACTORS[task]["width_reduction"]


# Preprocess label:
# 1) Read from path
# 2) Split by encoding grammar
# 3) Convert to integer if training
def preprocess_label(
    label_path: str, training: bool, w2i: Dict[str, int]
) -> Union[Tuple[List[int], int], List[str]]:
    label = open(label_path).read().split()
    if training:
        label = [w2i[w] for w in label]
        return label, len(label)
    return label


################################################################# CTC PREPROCESSING:


# CTC-loss preprocess function -> (xi, yi)
# xi[0] -> images, zero-padded to the maximum image width found
# xi[1] -> real width (after the CNN) of the images
# xi[2] -> labels, CTC-blank-padded to the maximum label length found
# xi[3] -> real length of the labels
# yi[0] -> dummy value for CTC-loss
def ctc_preprocess(
    images: list, labels: list, blank_index: int
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    # Unzip variables
    images, images_len = list(zip(*images))
    labels, labels_len = list(zip(*labels))
    # Obtain the current batch size
    num_samples = len(images)
    # Zero-pad images to maximum batch image width
    max_width = max(images, key=np.shape).shape[1]
    images = np.array(
        [
            np.pad(i, pad_width=((0, 0), (0, max_width - i.shape[1]), (0, 0)))
            for i in images
        ],
        dtype="float32",
    )
    images_len = np.array(images_len, dtype="int32").reshape((num_samples, 1))
    # CTC-blank-pad labels to maximum batch label length
    max_length = len(max(labels, key=len))
    labels = np.array(
        [i + [blank_index] * (max_length - len(i)) for i in labels], dtype="int32"
    )
    labels_len = np.array(labels_len, dtype="int32").reshape((num_samples, 1))
    # Format data
    xi = {
        "image": images,
        "image_len": images_len,
        "label": labels,
        "label_len": labels_len,
    }
    yi = {"ctc_loss": np.zeros(shape=(num_samples, 1), dtype="float32")}
    return xi, yi


# Train data generator
def train_data_generator(
    task: str,
    images_files: List[str],
    labels_files: List[str],
    w2i: Dict[str, int],
    batch_size: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    images_files, labels_files = shuffle(images_files, labels_files, random_state=42)

    size = len(images_files)
    start = 0
    while True:
        end = min(start + batch_size, size)
        images = [
            preprocess_image(task=task, image_path=i) for i in images_files[start:end]
        ]
        labels = [
            preprocess_label(label_path=i, training=True, w2i=w2i)
            for i in labels_files[start:end]
        ]
        xi, yi = ctc_preprocess(images, labels, blank_index=len(w2i))
        yield xi, yi

        if end == size:
            start = 0
            # Due to the current training set up (model.fit() is called one epoch at a time),
            # it does not make sense to shuffle the data in this step
            # as the generator is stopped after all the training data is seen
            # This is why it is important to shuffle the data at the very beginning,
            # so that at each epoch it is seen in a different order
            # Uncomment the following line, if we were to call model.fit() for longer that one epoch
            # images_files, labels_files = shuffle(
            #     images_files, labels_files, random_state=42
            # )
        else:
            start = end

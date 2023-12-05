from typing import Dict, List, Tuple

import swalign
from tensorflow import keras

from networks.metrics import compute_metrics
from my_utils.preprocessing import preprocess_label
from late_fusion.smith_waterman import (
    swalign_preprocess,
    undo_swalign_preprocess,
    dump,
    preprocess_prob,
    get_alignment,
)
from late_fusion.utils import get_predictions_and_probabilities


# Utility function for evaluating a multimodal transcription at prediction level approach over a dataset
def evaluate_multimodal_transcription(
    omr_model: keras.Model,
    amt_model: keras.Model,
    omr_image_files: List[str],
    amt_image_files: List[str],
    labels_files: List[str],
    i2w: Dict[int, str],
    match: int = 2,
    mismatch: int = -1,
    gap_penalty: int = -1,
) -> Tuple[float, float]:
    y_pred_comb_acc = []
    # Obtain predictions and their corresponding probability arrays for OMR task
    omr_y_pred_acc, omr_y_pred_prob_acc = get_predictions_and_probabilities(
        task="omr",
        model=omr_model,
        images_files=omr_image_files,
        i2w=i2w,
    )
    # Obtain predictions and their corresponding probability arrays for AMT task
    amt_y_pred_acc, amt_y_pred_prob_acc = get_predictions_and_probabilities(
        task="amt",
        model=amt_model,
        images_files=amt_image_files,
        i2w=i2w,
    )
    # Obtain the callable object of swalign library that contains the align() method that performs the alignment
    scoring = swalign.NucleotideScoringMatrix(match, mismatch)
    # Gap penalty designates scores for insertion or deletion
    sw = swalign.LocalAlignment(scoring, gap_penalty=gap_penalty)
    # Perform the multimodal combination at prediction level
    for r, r_prob, q, q_prob in zip(
        omr_y_pred_acc, omr_y_pred_prob_acc, amt_y_pred_acc, amt_y_pred_prob_acc
    ):
        # Prepare for swalign computation
        r, q, swa2w = swalign_preprocess(r, q)
        # Smith-Waterman local alignment -> ref, query
        alignment = sw.align(r, q)
        q, m, r = dump(alignment)
        # Fusion policy
        q_prob = preprocess_prob(q, q_prob)
        r_prob = preprocess_prob(r, r_prob)
        alignment = get_alignment(q, m, r, q_prob, r_prob)
        # Undo the swalign preprocess and append to accumulator variable
        y_pred_comb_acc.append(undo_swalign_preprocess(alignment, swa2w))
    # Obtain true labels
    y_true_acc = [preprocess_label(i, training=False, w2i=None) for i in labels_files]
    # Compute metrics
    symer, seqer = compute_metrics(y_true=y_true_acc, y_pred=y_pred_comb_acc)
    print(
        f"SymER (%): {symer:.2f}, SeqER (%): {seqer:.2f} - From {len(y_true_acc)} samples"
    )
    return symer, seqer

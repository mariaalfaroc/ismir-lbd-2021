# -*- coding: utf-8 -*-

import os, gc

from typing import Tuple

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import swalign

import config
from data_processing import get_folds_filenames, get_datafolds_filenames, preprocess_image, preprocess_label, load_dictionaries
from evaluation import compute_metrics

# Multimodal image and audio music transcription
# Carlos de la Fuente, Jose J. Valero-Mas, Francisco J. Castellanos, and Jorge Calvo-Zaragoza 

# OMR and AMT combination at prediction level:
# 1. Input the score image to the OMR model and the CQT image to the AMT model
# 2. Obtain predictions for each model
# 3. Apply alignment policy
    # 3.1. Greedy decoding policy
    # 3.2. Merge consecutive symbols (average)
    # 3.3. Remove CTC-blank symbols
    # 3.4. Smith-Waterman local alignment algorithm (swalign)
    # 3.5. Fusion policy
        # - Both sequences match on a token -> included
        # - Sequences disagree on a token -> include that of the highest probability
        # - A sequence misses a token -> include that of the other

# Utility function for merging consecutive repeated values in a sequence and averaging their correspoding probabilities
def merge_repeated(y_dec: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
def filter_out_value(y_dec: np.ndarray, y_prob: np.ndarray, blank_index: int) -> Tuple[np.ndarray, np.ndarray]:
    not_blank_ids = np.where(y_dec != blank_index)
    return y_dec[not_blank_ids], y_prob[not_blank_ids]

# CTC-greedy decoder (merge repeated elements, remove blank labels, and covert back to string labels)
# Returns such best path and its probability array
def manual_ctc_greedy_decoder(y_pred: np.ndarray, input_length: list, i2w: dict) -> Tuple[list, list]:
    # Obtain the best path (class of highest probability at each time step) for each batch item
    y_pred_best = np.argmax(y_pred, axis=-1)
    # Obtain the associated probability for each time step of the best path of each batch item
    y_pred_best_prob = np.amax(y_pred, axis=-1)
    # Merge consecutive repeated classes and filter out blank label class
    y_pred_dec = []
    y_pred_prob_dec = []
    for y, prob, length in zip(y_pred_best, y_pred_best_prob, input_length):
        y, prob = merge_repeated(y[:int(length)], prob[:int(length)])
        y, prob = filter_out_value(y, prob, blank_index=len(i2w))
        # i2w conversion
        y = [i2w[int(i)] for i in y]
        y_pred_dec.append(y)
        y_pred_prob_dec.append(list(prob))
    return y_pred_dec, y_pred_prob_dec

# --------------------

# Utility function for obtaining the predictions and their associated probability arrays of a dataset
def get_predictions_and_probabilities(model, images_files, i2w):
    y_pred_acc = []
    y_pred_prob_acc = []
    # Iterate over images in batches
    for start in range(0, len(images_files), config.batch_size):
        images, images_len = list(zip(*[preprocess_image(i) for i in images_files[start:start + config.batch_size]]))
        # Zero-pad images to maximum batch image width
        max_width = max(images, key=np.shape).shape[1]
        images = np.array([np.pad(i, pad_width=((0, 0), (0, max_width - i.shape[1]), (0, 0))) for i in images], dtype="float32")
        # Obtain predictions
        y_pred = model(images, training=False)
        # Manual CTC greedy decoder
        y_pred, y_pred_prob = manual_ctc_greedy_decoder(y_pred, images_len, i2w)
        y_pred_acc.extend(y_pred)
        y_pred_prob_acc.extend(y_pred_prob)
    return y_pred_acc, y_pred_prob_acc

# --------------------

swalign_vocab = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
"$", "%", "&", "/", "(", ")", "=", "?", "¿", "*", "<", ">", "+", "#", "{", "}", ";", ":", "^", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Utility function for converting a string sequence to a compatible swalign string
def swalign_preprocess(r, q):
    current_vocab = sorted(set(r + q))
    # print(f"Length of current vocabulary: {len(current_vocab)} vs. Length of swalign vocabulary: {len(swalign_vocab)}")
    assert len(current_vocab) < len(swalign_vocab)
    w2swa = dict(zip(current_vocab, swalign_vocab))
    swa2w = dict(zip(swalign_vocab, current_vocab))
    r = ["¡"] + [w2swa[i] for i in r] + ["!"]
    q = ["¡"] + [w2swa[i] for i in q] + ["!"]
    return "".join(r), "".join(q), swa2w

# This is a modified version of the original dump() method for the Alignment class of the swalign library
# We have modified it to obtain (in the following order) the query, the matches, and the reference sequences; all of them have the same length
# Matches is a string that contains either "|" if sequences match on a token, or "." if they disagree, 
# or " " if one of them misses a token (in this case the token "-" is included at such position in the corresponding sequence)
def dump(alignment) -> Tuple[str, str, str]:
    i = alignment.r_pos
    j = alignment.q_pos

    q = ''
    m = ''
    r = ''
    qlen = 0
    rlen = 0

    for count, op in alignment.cigar:
        if op == 'M':
            qlen += count
            rlen += count
            for k in range(count):
                q += alignment.orig_query[j]
                r += alignment.orig_ref[i]
                if alignment.query[j] == alignment.ref[i] or (alignment.wildcard and (alignment.query[j] in alignment.wildcard or alignment.ref[i] in alignment.wildcard)):
                    m += '|'
                else:
                    m += '.'
                i += 1
                j += 1
        elif op == 'D':
            rlen += count
            for k in range(count):
                q += '-'
                r += alignment.orig_ref[i]
                m += ' '
                i += 1
        elif op == 'I':
            qlen += count
            for k in range(count):
                q += alignment.orig_query[j]
                r += '-'
                m += ' '
                j += 1
        elif op == 'N':
            q += '-//-'
            r += '-//-'
            m += '    '

    while q and r and m:
        return q, m, r

# Utility function for adapting the probility sequence after the swalign computation to be able to obtain the final alignment
def preprocess_prob(s: str, prob: list):
    new_prob = prob.copy()
    count = 0
    for id, v in enumerate(s):
        if v == "¡" or v == "!":
            new_prob.insert(id + count, 1)
            count += 1
        elif v == "-":
            new_prob.insert(id + count, 0)
            count += 1
    return new_prob

# Utility function for obtaining the final alignment sequence based on the fixed fusion policy
def get_alignment(q: str, m: str, r: str, q_prob: list, r_prob: list) -> str:
    alignment = ""
    for qv, mv, rv, qv_prob, rv_prob in zip(q, m, r, q_prob, r_prob):
        # There are three possible scenarios:
        # 1) Both sequences match on a token (mv == "|", qv == rv) -> included
        # 2) Sequences disagree on a token (mv == ".", qv != rv) -> include that of the highest probability
        # 3) A sequence misses a token (mv == " ", (qv or rv) == "-")-> include that of the other
        if mv == "|":
            # Scenario 1
            assert qv == rv
            alignment += qv
        elif mv == ".":
            # Scenario 2
            assert qv != rv
            alignment += qv if qv_prob >= rv_prob else rv
        elif mv == " ":
            # Scenario 3
            assert qv == "-" or rv == "-"
            alignment += qv if rv == "-" else rv
    return alignment

# Utility function for undoing the swalign preprocess
def undo_swalign_preprocess(alignment: str, swa2w: dict):
    return [swa2w[i] for i in list(alignment) if i not in ("¡", "!")]

# --------------------

# Utility function for evaluating a multimodal transcription at prediction level approach over a dataset
def evaluate_multimodal_transcription(omr_model, amt_model, omr_image_files, amt_image_files, labels_files, i2w, match=2, mismatch=-1, gap_penalty=-1):
    y_pred_comb_acc = []
    # Obtain predictions and their corresponding probability arrays for OMR task
    config.set_task(value="omr")
    config.set_data_globals()
    config.set_arch_globals(batch=16)
    omr_y_pred_acc, omr_y_pred_prob_acc = get_predictions_and_probabilities(omr_model, omr_image_files, i2w)
    # Obtain predictions and their corresponding probability arrays for AMT task
    config.set_task(value="amt")
    config.set_data_globals()
    config.set_arch_globals(batch=4)
    amt_y_pred_acc, amt_y_pred_prob_acc = get_predictions_and_probabilities(amt_model, amt_image_files, i2w)
    # Obtain the callable object of swalign library that contains the align() method that performs the alignment
    scoring = swalign.NucleotideScoringMatrix(match, mismatch)
    # Gap penalty designates scores for insertion or deletion
    sw =  swalign.LocalAlignment(scoring, gap_penalty)
    # Perform the multimodal combination at prediction level
    for r, r_prob, q, q_prob in zip(omr_y_pred_acc, omr_y_pred_prob_acc, amt_y_pred_acc, amt_y_pred_prob_acc):
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
    symer, seqer = compute_metrics(y_true_acc, y_pred_comb_acc)
    print(f"SymER (%): {symer:.2f}, SeqER (%): {seqer:.2f} - From {len(y_true_acc)} samples")
    return symer, seqer

# --------------------

# Utility function for performing a k-fold cross-validation multimodal experiment on a single dataset
def k_fold_multimodal_experiment(match=[2], mismatch=[-1], gap_penalty=[-1]):
    keras.backend.clear_session()
    gc.collect()

    # ---------- PRINT EXPERIMENT DETAILS

    print("k-fold multimodal image and audio music transcription at prediction level experiment")
    print(f"Data used {config.base_dir.stem}")
    print(f"match={match}, mismatch={mismatch}, gap_penalty={gap_penalty}")

    # ---------- DATA COLLECTION

    test_folds_files = get_folds_filenames("test")
    # OMR
    config.set_task(value="omr")
    config.set_data_globals()
    omr_test_images_fnames, test_labels_fnames = get_datafolds_filenames(test_folds_files)
    # AMT
    config.set_task(value="amt")
    config.set_data_globals()
    amt_test_images_fnames = get_datafolds_filenames(test_folds_files)[0]

    # ---------- K-FOLD EVALUATION

    # Start the k-fold evaluation scheme
    # k = len(omr_test_images_fnames)
    # for i in range(k):
    for i in range(1):
        # With 'clear_session()' called at the beginning,
        # Keras starts with a blank state at each iteration
        # and memory consumption is constant over time.
        keras.backend.clear_session()
        gc.collect()

        print(f"Fold {i}")

        # Get the current fold data
        omr_test_images, amt_test_images, test_labels = omr_test_images_fnames[i], amt_test_images_fnames[i], test_labels_fnames[i]
        assert len(omr_test_images) == len(amt_test_images) == len(test_labels) 
        print(f"Test: {len(omr_test_images)}")

        # Load the current fold dictionary
        # Both models have the same fold vocabulary, so load one of them
        i2w = load_dictionaries(filepath=config.output_dir / "omr" / f"Fold{i}" / "w2i.json")[1]

        # Load the models
        omr_pred_model_filepath = config.output_dir / "omr" / f"Fold{i}" / "best_model.keras"
        amt_pred_model_filepath = config.output_dir / "amt" / f"Fold{i}" / "best_model.keras"
        omr_prediction_model = keras.models.load_model(omr_pred_model_filepath)
        amt_prediction_model = keras.models.load_model(amt_pred_model_filepath)

        # Set filepaths outputs
        output_dir = config.output_dir / "Smith-Waterman" / f"Fold{i}"
        os.makedirs(output_dir, exist_ok=True)
        log_path = output_dir / "logs.csv"

        symer_acc = []
        seqer_acc = []
        # Iterate over the different match, mismatch and gap_penalty values
        for m, mism, gp in zip(match, mismatch, gap_penalty):
            # Multimodal transcription evaluation
            symer, seqer = evaluate_multimodal_transcription(
                omr_model=omr_prediction_model, amt_model=amt_prediction_model, 
                omr_image_files=omr_test_images, amt_image_files=amt_test_images, labels_files=test_labels, 
                i2w=i2w, 
                match=m, mismatch=mism, gap_penalty=gp)
            symer_acc.append(symer)
            seqer_acc.append(seqer)
        # Save fold logs
        logs = {
            "match": match,
            "mismatch": mismatch,
            "gap_penalty": gap_penalty,
            "symer": symer_acc,
            "seqer": seqer_acc
        }
        logs = pd.DataFrame.from_dict(logs)
        logs.to_csv(log_path, index=False)

        # Clear memory
        del omr_test_images, amt_test_images, test_labels
        del omr_prediction_model, amt_prediction_model

    return

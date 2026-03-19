"""Early time-window evaluation for prefix-sensitive attack detection."""

import os
from collections import defaultdict

import pandas as pd

from pidsmaker.detection.evaluation_methods.evaluation_utils import (
    classifier_evaluation,
    datetime_to_ns_time_US_handle_nano,
    get_threshold,
    plot_precision_recall,
    plot_simple_scores,
    reduce_losses_to_score,
)
from pidsmaker.utils.utils import datetime_to_ns_time_US, listdir_sorted, log, log_tqdm


def _load_tw_results(test_tw_path, threshold_method, threshold):
    tw_to_losses = defaultdict(list)
    filelist = listdir_sorted(test_tw_path)

    for tw, file in enumerate(log_tqdm(sorted(filelist), desc="Compute labels")):
        file = os.path.join(test_tw_path, file)
        df = pd.read_csv(file).to_dict(orient="records")
        for line in df:
            tw_to_losses[tw].append(line["loss"])

    results = {}
    for tw, losses in tw_to_losses.items():
        pred_score = reduce_losses_to_score(losses, threshold_method)
        results[tw] = {"score": pred_score, "y_hat": int(pred_score > threshold)}
    return results, filelist


def _label_malicious_windows(results, tw_to_malicious_nodes):
    malicious_tws = set(tw_to_malicious_nodes.keys())
    for tw, result in results.items():
        result["y_true"] = int(tw in malicious_tws)
    return results


def _parse_tw_ranges(filelist):
    tw_ranges = []
    for file in filelist:
        start_str, end_str = file.split("~")
        tw_ranges.append(
            (
                datetime_to_ns_time_US_handle_nano(start_str),
                datetime_to_ns_time_US_handle_nano(end_str),
            )
        )
    return tw_ranges


def _attack_to_windows(cfg, tw_ranges):
    attack_windows = {}
    for attack_idx, attack_tuple in enumerate(cfg.dataset.attack_to_time_window):
        attack_start = datetime_to_ns_time_US(attack_tuple[1])
        attack_end = datetime_to_ns_time_US(attack_tuple[2])
        matched = []
        for tw_idx, (tw_start, tw_end) in enumerate(tw_ranges):
            overlaps = tw_start < attack_end and attack_start < tw_end
            if overlaps:
                matched.append(tw_idx)
        if matched:
            attack_windows[attack_idx] = matched
    return attack_windows


def _compute_early_metrics(results, attack_to_windows):
    ks = (1, 3, 5)
    delays = []
    detected = 0
    early_hits = {k: 0 for k in ks}

    for attack_idx, windows in attack_to_windows.items():
        onset = min(windows)
        detected_windows = [tw for tw in windows if results.get(tw, {}).get("y_hat", 0) == 1]
        if not detected_windows:
            log(f"Attack {attack_idx}: missed in malicious windows {windows}")
            continue

        first_detected = min(detected_windows)
        delay = first_detected - onset
        delays.append(delay)
        detected += 1

        for k in ks:
            if delay <= k - 1:
                early_hits[k] += 1

        log(
            f"Attack {attack_idx}: onset_tw={onset}, first_detected_tw={first_detected}, delay={delay}"
        )

    total_attacks = len(attack_to_windows)
    median_delay = float("nan")
    if delays:
        sorted_delays = sorted(delays)
        median_delay = sorted_delays[len(sorted_delays) // 2]

    stats = {
        "num_attacks": total_attacks,
        "detected_attacks": detected,
        "attack_detection_rate": round(detected / total_attacks, 5) if total_attacks else 0.0,
        "mean_onset_delay": round(sum(delays) / len(delays), 5) if delays else float("nan"),
        "median_onset_delay": round(median_delay, 5) if delays else float("nan"),
    }
    for k, hits in early_hits.items():
        stats[f"early_recall_at_{k}"] = round(hits / total_attacks, 5) if total_attacks else 0.0

    stats["adp_score"] = stats["early_recall_at_1"]
    stats["discrimination"] = stats["attack_detection_rate"]
    return stats


def main(val_tw_path, test_tw_path, model_epoch_dir, cfg, tw_to_malicious_nodes, **kwargs):
    out_dir = cfg.evaluation._precision_recall_dir
    os.makedirs(out_dir, exist_ok=True)

    threshold_method = cfg.evaluation.early_tw_evaluation.threshold_method
    threshold = get_threshold(val_tw_path, threshold_method)
    log(f"Threshold: {threshold:.3f}")

    results, filelist = _load_tw_results(test_tw_path, threshold_method, threshold)
    results = _label_malicious_windows(results, tw_to_malicious_nodes)

    tw_ranges = _parse_tw_ranges(filelist)
    attack_to_windows = _attack_to_windows(cfg, tw_ranges)

    ordered_results = [result for _, result in sorted(results.items())]
    y_truth = [result["y_true"] for result in ordered_results]
    y_preds = [result["y_hat"] for result in ordered_results]
    pred_scores = [result["score"] for result in ordered_results]

    pr_img_file = os.path.join(out_dir, f"pr_curve_{model_epoch_dir}.png")
    simple_scores_img_file = os.path.join(out_dir, f"simple_scores_{model_epoch_dir}.png")
    plot_precision_recall(pred_scores, y_truth, pr_img_file)
    plot_simple_scores(pred_scores, y_truth, simple_scores_img_file)

    stats = classifier_evaluation(y_truth, y_preds, pred_scores)
    stats.update(_compute_early_metrics(results, attack_to_windows))
    return stats

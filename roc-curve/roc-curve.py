import numpy as np

def roc_curve(y_true, y_score):
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # ── Step 1: sort by descending score (ties: put positives last → conservative) ──
    # np.lexsort: primary key = y_true (ascending), secondary = y_score (ascending)
    # then reverse → descending score; among ties, label=0 comes before label=1
    desc_idx      = np.lexsort((y_true, y_score))[::-1]
    sorted_labels = y_true[desc_idx]
    sorted_scores = y_score[desc_idx]

    # ── Step 2: cumulative TP and FP as we lower the threshold ──
    cum_tp = np.cumsum(sorted_labels)           # running true  positives
    cum_fp = np.cumsum(1 - sorted_labels)       # running false positives

    total_p = cum_tp[-1]                        # total real positives
    total_n = cum_fp[-1]                        # total real negatives

    # ── Step 3: keep only the LAST index of each unique score (group ties) ──
    # diff finds where score changes; last index of each group = just before change
    distinct_mask = np.concatenate(
        (np.diff(sorted_scores) != 0, [True])   # True at last element of each group
    )

    tp = cum_tp[distinct_mask]
    fp = cum_fp[distinct_mask]
    thresholds = sorted_scores[distinct_mask]

    # ── Step 4: compute rates ──
    tpr = tp / total_p
    fpr = fp / total_n

    # ── Step 5: prepend the (0, 0, inf) starting point ──
    tpr        = np.concatenate(([0.0], tpr))
    fpr        = np.concatenate(([0.0], fpr))
    thresholds = np.concatenate(([np.inf], thresholds))

    return fpr, tpr, thresholds
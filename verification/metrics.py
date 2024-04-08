import numpy
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score, accuracy_score
from eer import eer
import numpy as np
import bisect


def _generate_roc(similarities, labels):
    fpr, tpr, thresholds = roc_curve(labels, similarities, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc


def _get_best_threshold(fpr, tpr, thresholds):
    return thresholds[np.nanargmin(np.abs(fpr - (1 - tpr)))]


def _similarity(embedding1, embedding2):
    embedding1_magnitude = numpy.linalg.norm(embedding1)
    embedding2_magnitude = numpy.linalg.norm(embedding2)
    cosine_similarity = numpy.dot(embedding1, embedding2) / (embedding1_magnitude * embedding2_magnitude)
    return cosine_similarity


def _run_stats(model, dataloader, device):
    similarities = []
    labels = []
    model.eval()

    for idx, data_sample in enumerate(dataloader):
        img1 = data_sample["img1"].to(device)
        img2 = data_sample["img2"].to(device)
        label = data_sample["label"]

        with torch.no_grad():
            embedding1 = model(img1)["embedding"].cpu()
            embedding2 = model(img2)["embedding"].cpu()

        similarity = [_similarity(em1, em2) for em1, em2 in zip(embedding1, embedding2)]
        similarities.extend(similarity)
        labels.extend(label)

    return similarities, labels

def _preds(similarities, best_threshold):
    return [1 if sim > best_threshold else 0 for sim in similarities]

def _f1_accuracy_eer(similarities, labels, best_threshold):
    preds = _preds(similarities, best_threshold)
    f1 = f1_score(labels, preds)
    accuracy = accuracy_score(labels, preds)
    eer_score = eer(similarities, labels)
    return f1, accuracy, eer_score


def detailed_roc_curve(labels, scores, num_thresholds=10000):
    thresholds = np.linspace(min(scores), max(scores), num_thresholds)
    tpr = []
    fpr = []

    for thresh in thresholds:
        # Predicted positives/negatives
        pred_pos = scores >= thresh
        pred_neg = ~pred_pos

        # True positives and false positives
        tp = np.sum([label == 1 and pred for label, pred in zip(labels, pred_pos)])
        fp = np.sum([label == 0 and pred for label, pred in zip(labels, pred_pos)])

        # False negatives and true negatives
        fn = np.sum([label == 1 and pred for label, pred in zip(labels, pred_neg)])
        tn = np.sum([label == 0 and pred for label, pred in zip(labels, pred_neg)])

        _tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
        _fpr = fp / (fp + tn) if (fp + tn) != 0 else 0

        # Calculate TPR and FPR
        tpr.append(_tpr)
        fpr.append(_fpr)

    return np.array(fpr), np.array(tpr), thresholds


def test_and_print(model, dataloader, device, output_dir, index):
    similarities, labels = _run_stats(model, dataloader, device)
    fpr, tpr, thresholds, roc_auc = _generate_roc(similarities, labels)
    best_threshold = _get_best_threshold(fpr, tpr, thresholds)

    _print_all_tar_at_far(fpr, tpr)
    _print_stats(similarities, labels, best_threshold)

    _plot_far_frr(fpr, tpr, thresholds, best_threshold, os.path.join(output_dir, "far_frr-" + str(index) + ".png"))
    fpr, tpr, thresholds = detailed_roc_curve(labels, similarities)
    _plot_tpr_fpr(tpr, fpr, os.path.join(output_dir, "tpr_fpr-" + str(index) + ".png"))
    _plot_accuracy_progress(similarities, labels, best_threshold, os.path.join(output_dir, "accuracy-" + str(index) + ".png"))


def _plot_accuracy_progress(similarities, labels, best_threshold, path):
    accuracies = []
    thresholds = [i / 1000 for i in range(1, 1001)]

    for threshold in thresholds:
        preds = _preds(similarities, threshold)
        accuracies.append(accuracy_score(labels, preds))

    print("MAX ACCURACY:", max(accuracies), "at threshold:", thresholds[np.argmax(accuracies)])
    plt.plot(thresholds, accuracies, label='Accuracy')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.axvline(x=best_threshold, label='Optimal threshold value', color='red')
    plt.legend()
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.grid(True)
    plt.savefig(path)
    plt.close()

def _plot_tpr_fpr(tpr, fpr, path):
    plt.plot(fpr, tpr)
    # plt.axvline(x=1e-2, color="red", label="Testing threshold")
    # plt.axvline(x=1e-3, color="red", label="Testing threshold")
    # plt.axvline(x=1e-4, color="red", label="Testing threshold")
    # plt.axvline(x=1e-5, color="red", label="Testing threshold")
    # plt.axvline(x=1e-6, color="red", label="Testing threshold")
    # plt.axvline(x=1e-7, color="red", label="Testing threshold")
    plt.xlabel('FPR (False Positive Rate)')
    plt.ylabel('TPR (True Positive Rate)')
    plt.xscale('log')
    plt.ylim([0.9, 1])
    plt.xlim([1e-10, 1])
    plt.grid(True)
    plt.savefig(path)
    plt.close()


def _plot_far_frr(fpr, tpr, thresholds, best_threshold, path):
    far = fpr
    frr = 1 - tpr

    plt.plot(thresholds, far, label='FAR (False Acceptance Rate)')
    plt.plot(thresholds, frr, label='FRR (False Rejection Rate)')
    plt.axvline(x=best_threshold, label='Optimal threshold value', color='red')
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.legend()
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.title('FAR and FRR curves for EER')
    plt.grid(True)
    plt.savefig(path)
    plt.close()


def _print_stats(similarities, labels, best_threshold):
    f1, accuracy, eer_score = _f1_accuracy_eer(similarities, labels, best_threshold)
    print("Optimal F1:", f1)
    print("Optimal Accuracy:", accuracy)
    print("EER:", eer_score)
    print("Optimal threshold:", best_threshold)


def _print_tar_at_far(fpr, tpr, far_rate):
    index = bisect.bisect_left(fpr, far_rate) - 1
    print("> TAR(%)@FAR=" + str(far_rate) + ":", tpr[index])


def _print_all_tar_at_far(fpr, tpr):
    for i in range(2, 7):
        _print_tar_at_far(fpr, tpr, 10 ** (-i))



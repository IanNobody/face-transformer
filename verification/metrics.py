import gc
import glob

import numpy
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score, accuracy_score
from eer import eer
import numpy as np
import bisect
import random

from pytorch_metric_learning.distances.cosine_similarity import CosineSimilarity
from models.lightning_wrapper import LightningWrapper

DETAILED = False
#SEED = 69
SEED = 69
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

distance = CosineSimilarity()

def _generate_roc(similarities, labels):
    fpr, tpr, thresholds = roc_curve(labels, similarities, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc


def _get_best_threshold(fpr, tpr, thresholds):
    return thresholds[np.nanargmin(np.abs(fpr - (1 - tpr)))]


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

        similarity = [distance(em1.unsqueeze(0), em2.unsqueeze(0))[0].item() for em1, em2 in zip(embedding1, embedding2)]
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


def _detailed_roc_curve(labels, scores, num_thresholds=10000):
    thresholds = np.linspace(min(scores), max(scores), num_thresholds)
    tpr = []
    fpr = []

    for thresh in thresholds:
        pred_pos = scores >= thresh
        pred_neg = ~pred_pos

        tp = np.sum([label == 1 and pred for label, pred in zip(labels, pred_pos)])
        fp = np.sum([label == 0 and pred for label, pred in zip(labels, pred_pos)])

        fn = np.sum([label == 1 and pred for label, pred in zip(labels, pred_neg)])
        tn = np.sum([label == 0 and pred for label, pred in zip(labels, pred_neg)])

        _tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
        _fpr = fp / (fp + tn) if (fp + tn) != 0 else 0

        tpr.append(_tpr)
        fpr.append(_fpr)

    return np.array(fpr), np.array(tpr), thresholds


def test_dir(dir, config, model, number_of_classes, eval_dataloader, output_dir):
    device = torch.device("cuda:" + str(config.devices[0]))
    f1s, accuracys, max_accuracys, eer_scores, thresholds = [], [], [], [], []

    if dir.endswith(".pth") or dir.endswith(".ckpt"):
        checkpoint_files = [dir]
    else:
        pattern_ckpt = os.path.join(dir, "**", '*.ckpt')
        pattern_pth = os.path.join(dir, "**", '*.pth')
        checkpoint_files = sorted(glob.glob(pattern_ckpt, recursive=True) + glob.glob(pattern_pth, recursive=True))

    for idx, ckpt_file in enumerate(checkpoint_files):
        print("----------------------------")
        print("Checking file: ", ckpt_file)

        if config.old_checkpoint_format:
            model.load_backbone_weights(ckpt_file)
            lightning_model = LightningWrapper(
                model=model,
                config=config,
                num_classes=number_of_classes
            )
        else:
            lightning_model = LightningWrapper.load_from_checkpoint(
                ckpt_file, model=model,
                config=config,
                num_classes=number_of_classes,
                map_location=device,
                strict=False
            )

        lightning_model.to(device)
        f1, accuracy, max_accuracy, eer_score, threshold = test_and_print(lightning_model, eval_dataloader, device, output_dir, idx)
        f1s.append(f1)
        accuracys.append(accuracy)
        max_accuracys.append(max_accuracy)
        eer_scores.append(eer_score)
        thresholds.append(threshold)

        del lightning_model
        gc.collect()
        torch.cuda.empty_cache()

    _plot_stats(f1s, max_accuracys, eer_scores, thresholds, output_dir)
    return f1s, accuracys, max_accuracys, eer_scores, thresholds

def test_and_print(model, dataloader, device, output_dir, index):
    similarities, labels = _run_stats(model, dataloader, device)
    fpr, tpr, thresholds, roc_auc = _generate_roc(similarities, labels)
    best_threshold = _get_best_threshold(fpr, tpr, thresholds)
    f1, accuracy, eer_score = _f1_accuracy_eer(similarities, labels, best_threshold)

    _print_all_tar_at_far(fpr, tpr, thresholds)
    _print_stats(f1, accuracy, eer_score, best_threshold)

    _plot_far_frr(fpr, tpr, thresholds, eer_score, os.path.join(output_dir, "far_frr-" + str(index) + ".png"))
    if DETAILED:
        fpr, tpr, thresholds = _detailed_roc_curve(labels, similarities)
    _plot_roc(tpr, fpr, os.path.join(output_dir, "roc-" + str(index) + ".png"))
    max_accuracy = _plot_accuracy_progress(similarities, labels, os.path.join(output_dir, "accuracy-" + str(index) + ".png"))

    return f1, accuracy, max_accuracy, eer_score, best_threshold


def _plot_stats(f1, max_accuracy, eer_score, threshold, path):
    plt.title("F1 score")
    plt.plot(f1)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.ylim([0.5, 1])
    plt.savefig(os.path.join(path, "f1.png"))
    plt.close()

    plt.title("Accuracy")
    plt.plot(max_accuracy)
    plt.axhline(y=max(max_accuracy), label='Maximal accuracy (' + '{:.3f}'.format(max(max_accuracy)) + ')',
                linestyle=':', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.ylim([0.5, 1])
    plt.legend()
    plt.savefig(os.path.join(path, "acc.png"))
    plt.close()

    plt.title("Equal Error Rate")
    plt.plot(eer_score)
    plt.xlabel('Epoch')
    plt.ylabel('EER')
    plt.ylim([0, 0.4])
    plt.savefig(os.path.join(path, "eer.png"))
    plt.close()

    plt.title("EER threshold")
    plt.plot(threshold)
    plt.xlabel('Epoch')
    plt.ylabel('Threshold')
    # plt.ylim([0, 1])
    plt.savefig(os.path.join(path, "threshold.png"))
    plt.close()


def _plot_accuracy_progress(similarities, labels, path):
    num_thresholds = 1000 if DETAILED else 100
    accuracies = []
    thresholds = [i / num_thresholds for i in range(1, num_thresholds + 1)]

    for threshold in thresholds:
        preds = _preds(similarities, threshold)
        accuracies.append(accuracy_score(labels, preds))

    max_acc = max(accuracies)
    print("MAX ACCURACY:", max_acc, "at threshold:", thresholds[np.argmax(accuracies)])
    plt.title("Accuracy/Threshold curve")
    plt.plot(thresholds, accuracies, label='Accuracy')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.axhline(y=max_acc, label='Maximal accuracy (' + '{:.3f}'.format(max_acc) + ')',
                linestyle=':', color='red')
    plt.legend()
    plt.ylim([0.5, 1])
    plt.xlim([0, 1])
    plt.grid(True)
    plt.savefig(path)
    plt.close()
    return max_acc


def _plot_roc(tpr, fpr, path):
    plt.title("ROC curve")
    plt.plot(fpr, tpr)
    plt.xlabel('FPR (False Positive Rate)')
    plt.ylabel('TPR (True Positive Rate)')
    plt.xscale('log')
    plt.ylim([0.9, 1])
    plt.xlim([1e-10, 1])
    plt.grid(True)
    plt.savefig(path)
    plt.close()


def _plot_far_frr(fpr, tpr, thresholds, eer, path):
    far = fpr
    frr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.abs(far - frr))]

    plt.plot(thresholds, far, label='FAR (False Acceptance Rate)')
    plt.plot(thresholds, frr, label='FRR (False Rejection Rate)')
    plt.plot(eer_threshold, eer, label='EER (' + '{:.3f}'.format(eer) + ')', color='red', marker='o')
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.legend()
    plt.xlim([0, 1])
    plt.title('FAR and FRR curves for EER')
    plt.grid(True)
    plt.savefig(path)
    plt.close()


def _print_stats(f1, accuracy, eer_score, best_threshold):
    print("Optimal F1:", f1)
    print("Optimal Accuracy:", accuracy)
    print("EER:", eer_score)
    print("Optimal threshold:", best_threshold)


def _print_tar_at_far(fpr, tpr, thresholds, far_rate):
    index = bisect.bisect_left(fpr, far_rate) - 1
    print("> TAR(%)@FAR=" + str(far_rate) + ":", tpr[index], " at threshold:", thresholds[index])


def _print_all_tar_at_far(fpr, tpr, thresholds):
    for i in range(2, 7):
        _print_tar_at_far(fpr, tpr, thresholds, 10 ** (-i))



import numpy
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import re
import os
from checkpointing.checkpoint import load_weights
from eer import eer
import numpy as np


class Metrics:
    def __init__(self, model, dataloader, config):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.similarities = []
        self.labels = []
        self.stats = { "threshold": [], "roc_auc": [], "f1": [], "accuracy": [], "eer": [], "tpr": [], "fpr": [], "precision": [], "recall": [] }

    def validate(self):
        self._run_stats()
        fpr, tpr, thresholds, roc_auc = self._generate_roc()
        best_threshold = self._get_best_threshold(fpr, tpr, thresholds, 0.0001)
        stats = self._threshold_based_statistics(best_threshold)
        f1 = self._compute_f1(stats)
        accuracy = (stats["true_positive"] + stats["true_negative"]) / len(self.similarities)
        return f1, accuracy

    def _run_stats(self):
        self.model.eval()
        self._clear_stats()

        for idx, data_sample in enumerate(self.dataloader):
            img1 = data_sample["img1"].to(self.config.device)
            img2 = data_sample["img2"].to(self.config.device)
            label = data_sample["label"]

            with torch.no_grad():
                if self.config.model_name == "dat":
                    embedding1 = self.model(img1)[0].cpu()
                    embedding2 = self.model(img2)[0].cpu()
                else:
                    embedding1 = self.model(img1).cpu()
                    embedding2 = self.model(img2).cpu()

                similarity = [self._similarity(em1, em2) for em1, em2 in zip(embedding1, embedding2)]
                self.similarities.extend(similarity)
                self.labels.extend(label)

    def _clear_stats(self):
        self.similarities = []
        self.labels = []

    def test_all_weights(self, checkpoint_path, output_dir, model):
        if not os.path.isdir(checkpoint_path) and not os.path.isfile(checkpoint_path):
            print("Invalid checkpoint path.")
            exit(-1)

        checkpoint_pattern = re.compile(r'^checkpoint-(\d+)\.pth$')
        processed_files = set()

        while True:
            # try:
                files = sorted(os.listdir(checkpoint_path)) if os.path.isdir(checkpoint_path) else [checkpoint_path]
                prev_len = len(processed_files)

                for file in files:
                    match = checkpoint_pattern.match(file)
                    if match:
                        checkpoint_number = int(match.group(1))

                        if checkpoint_number not in processed_files:
                            file_path = os.path.join(checkpoint_path, file)
                            load_weights(model, file_path, self.config.device)
                            print("**** Weight from epoch", checkpoint_number, "****")

                            self._clear_stats()
                            self._run_stats()

                            fpr, tpr, thresholds, roc_auc = self._generate_roc()
                            best_threshold = self._get_best_threshold(fpr, tpr, thresholds, 0.0001)

                            filepath = os.path.join(output_dir, "roc-" + str(checkpoint_number) + ".png")
                            if filepath:
                                self._save_roc(fpr, tpr, roc_auc, filepath)

                            stats = self._threshold_based_statistics(best_threshold)
                            self._append_stats(stats, best_threshold, roc_auc)

                            processed_files.add(checkpoint_number)

                if len(processed_files) == prev_len:
                    self._print_stats()
                    break
            # except Exception as e:
            #     print("Error occurred while processing files. \nError description: ", e)
            #     exit(-1)

    def test_and_print(self, output_dir):
        self._run_stats()
        fpr, tpr, thresholds, roc_auc = self._generate_roc()
        best_threshold = self._get_best_threshold(fpr, tpr, thresholds, 0.0001)
        filepath = os.path.join(output_dir, "roc.png")
        self._save_roc(fpr, tpr, roc_auc, filepath)
        stats = self._threshold_based_statistics(best_threshold)
        self._append_stats(stats, best_threshold, roc_auc)
        self._print_stats()

    def _append_stats(self, stats, threshold, roc_auc):
        self.stats["threshold"].append(threshold)
        self.stats["roc_auc"].append(roc_auc)
        self.stats["f1"].append(self._compute_f1(stats))
        self.stats["accuracy"].append((stats["true_positive"] + stats["true_negative"]) / len(self.similarities))
        self.stats["eer"].append(self._get_eer())
        self.stats["tpr"].append(stats["true_positive"] / (stats["true_positive"] + stats["false_negative"]))
        self.stats["fpr"].append(stats["false_positive"] / (stats["false_positive"] + stats["true_negative"]))
        self.stats["precision"].append(stats["true_positive"] / (stats["true_positive"] + stats["false_positive"]))
        self.stats["recall"].append(stats["true_positive"] / (stats["true_positive"] + stats["false_negative"]))

    def _print_stats(self):
        print("-------- Thresholds --------")
        print(self.stats["threshold"])
        print("------------ AUC ------------")
        print(self.stats["roc_auc"])
        print("------------ F1 ------------")
        print(self.stats["f1"])
        print("--------- Accuracy ---------")
        print(self.stats["accuracy"])
        print("------------ EER ------------")
        print(self.stats["eer"])
        print("------------ TPR ------------")
        print(self.stats["tpr"])
        print("------------ FPR ------------")
        print(self.stats["fpr"])
        print("--------- Precision ---------")
        print(self.stats["precision"])
        print("----------- Recall -----------")
        print(self.stats["recall"])
        print("------------ BEST ------------")
        print("Max F1: ", max(self.stats["f1"]))
        print("Max Accuracy: ", max(self.stats["accuracy"]))
        print("Min EER: ", min(self.stats["eer"]))
        print("------------------------------")

    def _generate_roc(self):
        fpr, tpr, thresholds = roc_curve(self.labels, self.similarities)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, thresholds, roc_auc

    def _get_eer(self):
        return eer(self.similarities, self.labels)

    @staticmethod
    def _get_best_threshold(fpr, tpr, thresholds, max_t):
        # indexes = [True if rate <= max_t else False for rate in fpr]
        # best_index = numpy.argmax(tpr[indexes])
        # return thresholds[indexes][best_index]
        return thresholds[np.nanargmin(np.abs(fpr - (1 - tpr)))]

    @staticmethod
    def _save_roc(fpr, tpr, roc_auc, path):
        plt.figure(figsize=(12, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve')
        plt.legend(loc='lower right')
        plt.savefig(path)

    def _threshold_based_statistics(self, threshold):
        stats = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}

        for idx in range(len(self.similarities)):
            if self.similarities[idx] > threshold:
                if self.labels[idx]:
                    stats["true_positive"] += 1
                else:
                    stats["false_positive"] += 1
            else:
                if self.labels[idx]:
                    stats["false_negative"] += 1
                else:
                    stats["true_negative"] += 1

        return stats

    @staticmethod
    def _similarity(embedding1, embedding2):
        embedding1_magnitude = numpy.linalg.norm(embedding1)
        embedding2_magnitude = numpy.linalg.norm(embedding2)
        cosine_similarity = numpy.dot(embedding1, embedding2) / (embedding1_magnitude * embedding2_magnitude)
        return cosine_similarity

    @staticmethod
    def _compute_f1(stats):
        true_pos = stats["true_positive"]
        false_pos = stats["false_positive"]
        false_neg = stats["false_negative"]

        precision_divisor = true_pos + false_pos
        precision = true_pos / precision_divisor if precision_divisor > 0 else 0
        recall_divisor = true_pos + false_neg
        recall = true_pos / recall_divisor if recall_divisor > 0 else 0
        return 2 * ((precision * recall) / (precision + recall)) if precision + recall > 0 else 0

import numpy
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import re
import os
from checkpointing.checkpoint import load_weights
from eer import eer
import numpy as np
import bisect


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
                elif self.config.model_name == "multitask_openclip":
                    embedding1 = self.model(img1)["embedding"].cpu()
                    embedding2 = self.model(img2)["embedding"].cpu()
                else:
                    embedding1 = self.model(img1)["embedding"].cpu()
                    embedding2 = self.model(img2)["embedding"].cpu()

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
                            self._print_all_tar_at_far(fpr, tpr)
                            best_threshold = self._get_best_threshold(fpr, tpr, thresholds, 0.0001)

                            filepath = os.path.join(output_dir, "roc-" + str(checkpoint_number) + ".png")
                            if filepath:
                                self._save_roc(fpr, tpr, thresholds, roc_auc, filepath)

                            stats = self._threshold_based_statistics(best_threshold)
                            self._append_stats(stats, best_threshold, roc_auc)

                            processed_files.add(checkpoint_number)

                if len(processed_files) == prev_len:
                    self._print_stats()
                    break
            # except Exception as e:
            #     print("Error occurred while processing files. \nError description: ", e)
            #     exit(-1)

    def test_and_print(self, output_dir, index):
        self._run_stats()
        fpr, tpr, thresholds, roc_auc = self._generate_roc()
        self._print_all_tar_at_far(fpr, tpr)
        best_threshold = self._get_best_threshold(fpr, tpr, thresholds, 0.0001)
        filepath = os.path.join(output_dir, "roc-" + str(index) + ".png")
        self._save_roc(fpr, tpr, thresholds, roc_auc, filepath)
        self._save_tpr_fpr(tpr, fpr, os.path.join(output_dir, "tpr_fpr-" + str(index) + ".png"))
        stats = self._threshold_based_statistics(best_threshold)
        self._append_stats(stats, best_threshold, roc_auc)
        #self._compute_print(self.similarities, self.labels)
        self._print_stats()
        self.plot_accuracy_progress(self.similarities, self.labels, index)
        self.plot_tar_progress(self.similarities, self.labels, index)

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
        print("Threshold:", self.stats["threshold"])
        print("F1: ", self.stats["f1"])
        print("Accuracy: ", self.stats["accuracy"])
        print("EER: ", self.stats["eer"])

    def _generate_roc(self):
        fpr, tpr, thresholds = roc_curve(self.labels, self.similarities)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, thresholds, roc_auc

    def _get_eer(self):
        return eer(self.similarities, self.labels)

    def _print_tar_at_far(self, fpr, tpr, far_rate):
        index = bisect.bisect_left(fpr, far_rate) - 1
        length = fpr[index + 1] - fpr[index]
        diff = far_rate - fpr[index]
        interpol_coeff = diff / length
        interpol = tpr[index] + interpol_coeff * (tpr[index + 1] - tpr[index])
        print("> TAR(%)@FAR=" + str(far_rate) + ":", tpr[index], " (interpolated: " + str(interpol) + ")")

    def compute_tar_at_far(genuine_scores, impostor_scores, target_FAR):
        # Combine scores and label them (1 for genuine, 0 for impostor)
        scores = [(score, 1) for score in genuine_scores] + [(score, 0) for score in impostor_scores]

        # Sort scores in descending order
        scores.sort(key=lambda x: x[0], reverse=True)

        # Calculate FAR and TAR for each threshold
        FARs = []
        TARs = []
        total_genuine = len(genuine_scores)
        total_impostor = len(impostor_scores)
        TP = 0
        FP = 0

        for score, label in scores:
            if label == 1:
                TP += 1
            else:
                FP += 1

            FAR = FP / total_impostor
            TAR = TP / total_genuine

            FARs.append(FAR)
            TARs.append(TAR)

        # Find the closest FAR to the target and its corresponding TAR
        closest_FAR_index = min(range(len(FARs)), key=lambda i: abs(FARs[i] - target_FAR))
        return FARs[closest_FAR_index], TARs[closest_FAR_index]

    def plot_accuracy_progress(self, similarities, true_labels, index):
        # Calculate the number of positive and negative samples
        num_positives = sum(true_labels)
        num_negatives = len(true_labels) - num_positives

        # Sort the similarities and true labels in descending order
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_similarities = np.array(similarities)[sorted_indices]
        sorted_labels = np.array(true_labels)[sorted_indices]

        # Initialize lists to store accuracy and thresholds
        accuracy_list = []
        thresholds_list = []

        # Iterate through thresholds from 0.01 to 0.99 by step of 0.01
        for threshold in np.arange(0.01, 1, 0.01):
            # Compute predictions based on the current threshold
            predictions = (sorted_similarities >= threshold).astype(int)

            # Calculate accuracy
            accuracy = np.sum(predictions == sorted_labels) / len(predictions)

            # Append accuracy and threshold to the lists
            accuracy_list.append(accuracy)
            thresholds_list.append(threshold)

        # Plot accuracy progress
        plt.plot(thresholds_list, accuracy_list)
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Progress')
        plt.grid(True)
        plt.savefig('accuracy_progress-' + str(index) + '.png')
        plt.close()

    def calculate_far(self, similarities, labels, threshold):
        # Calculate the number of false acceptances
        false_acceptances = np.sum((similarities >= threshold) & (labels == 0))
        # Calculate the total number of impostor trials
        total_impostor_trials = np.sum(labels == 0)  # Count the number of impostor instances
        # Calculate the False Acceptance Rate (FAR)
        far = false_acceptances / total_impostor_trials
        return far

    def find_threshold_for_far(self, similarities, labels, target_far):
        # Sort similarities in descending order
        sorted_similarities = np.sort(similarities)[::-1]
        # Iterate through sorted similarities and find the threshold corresponding to target FAR
        for similarity in sorted_similarities:
            far = self.calculate_far(similarities, labels, similarity)
            if far <= target_far:
                return similarity
        # If no threshold found for the target FAR, return None
        return None

    def plot_tar_progress(self, similarities, true_labels, index):
        # Calculate the number of positive and negative samples
        num_positives = sum(true_labels)
        num_negatives = len(true_labels) - num_positives

        # Sort the similarities and true labels in descending order
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_similarities = np.array(similarities)[sorted_indices]
        sorted_labels = np.array(true_labels)[sorted_indices]

        # Initialize lists to store TAR scores for each threshold
        tar_scores = []

        # Iterate through thresholds from 0.01 to 0.99 by step of 0.01
        thresholds = np.arange(0.01, 1, 0.01)
        for threshold in thresholds:
            # Compute predictions based on the current threshold
            predictions = (sorted_similarities >= threshold).astype(int)

            # Calculate True Acceptance Rate (TAR)
            tar = np.sum((predictions == 1) & (sorted_labels == 1)) / num_positives

            # Append TAR to the list
            tar_scores.append(tar)

        # Plot TAR progress
        plt.plot(thresholds, tar_scores)
        plt.xlabel('Threshold')
        plt.ylabel('True Acceptance Rate (TAR)')
        plt.title('TAR Progress with Threshold Variation')
        plt.grid(True)
        plt.savefig('tar_progress-' + str(index) + '.png')
        plt.close()

    def _print_all_tar_at_far(self, fpr, tpr):
        for i in range(2, 7):
            self._print_tar_at_far(fpr, tpr, 10 ** (-i))

    @staticmethod
    def _get_best_threshold(fpr, tpr, thresholds, max_t):
        # indexes = [True if rate <= 0.001 else False for rate in fpr]
        # indexes = [True if rate <= max_t else False for rate in fpr]
        # best_index = numpy.argmax(tpr[indexes])
        # return thresholds[indexes][best_index]
        return thresholds[np.nanargmin(np.abs(fpr - (1 - tpr)))]

    @staticmethod
    def _save_roc(fpr, tpr, thresholds, roc_auc, path):
        eer_threshold = thresholds[np.argmin(np.abs(fpr - (1 - tpr)))]
        print("EER Threshold: ", eer_threshold)
        print("EER Accuracy: ", 1 - (fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))] + (1 - tpr[np.nanargmin(np.abs(fpr - (1 - tpr)))])) / 2)

        # Calculate FAR and FRR
        far = fpr
        frr = 1 - tpr
        accuracy = [1 - (ar + rr) / 2 for ar, rr in zip(far, frr)]

        # Plot FAR and FRR curves
        plt.plot(thresholds, far, label='FAR (False Acceptance Rate)')
        plt.plot(thresholds, frr, label='FRR (False Rejection Rate)')
        plt.plot(thresholds, accuracy, label='Accuracy')
        plt.xlabel('Threshold')
        plt.ylabel('Rate')
        plt.yscale('log')
        plt.ylim([1e-10, 1])
        plt.legend()
        plt.title('FAR and FRR curves for EER')
        plt.grid(True)
        plt.savefig(path)
        plt.close()

    def _save_tpr_fpr(self, tpr, fpr, path):
        plt.plot(tpr, fpr)
        plt.xlabel('TPR')
        plt.ylabel('FPR')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(path)
        plt.close()

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

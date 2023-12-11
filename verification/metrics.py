import numpy
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import re
import os
from checkpointing.checkpoint import load_weights


class Metrics:
    def __init__(self, model, dataloader, config):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.similarities = []
        self.labels = []

    def _run_stats(self):
        self.model.eval()

        l_embed = []
        l_entity = []

        for idx, data_sample in enumerate(self.dataloader):
            image = data_sample["image"].to(self.config.device)
            entity = data_sample["entity"].to(self.config.device)

            with torch.no_grad():
                out_embeddings = self.model(image)

            if self.config.model_name == "dat":
                out_embeddings = out_embeddings[0]

            if idx % 2 == 0:
                l_embed = out_embeddings
                l_entity = entity

                if idx == len(self.dataloader) - 1:
                    self._compare_asymmetric_batch(
                        {"embed": l_embed.cpu(), "entity": l_entity.cpu()}
                    )

            if idx % 2 == 1:
                self._compare_batches(
                    {"embed": l_embed.cpu(), "entity": l_entity.cpu()},
                    {"embed": out_embeddings.cpu(), "entity": entity.cpu()}
                )

    def _compare_asymmetric_batch(self, batch):
        if len(batch["embed"]) % 2 != 0:
            batch = {"embed": batch["embed"][:-1], "entity": batch["entity"][:-1]}

        half_idx = len(batch["embed"]) // 2

        b1 = {"embed": batch["embed"][:half_idx], "entity": batch["entity"][:half_idx]}
        b2 = {"embed": batch["embed"][half_idx:], "entity": batch["entity"][half_idx:]}
        return self._compare_batches(b1, b2)

    def _compare_batches(self, batch1, batch2):
        if len(batch1["embed"]) == len(batch2["embed"]):
            for idx in range(len(batch1["embed"])):
                self.similarities.append(self._similarity(batch1["embed"][idx], batch2["embed"][idx]))
                self.labels.append(batch1["entity"][idx] == batch2["entity"][idx])
        else:
            bigger_batch = batch1 if len(batch1["embed"]) > len(batch2["embed"]) else batch2
            smaller_batch = batch1 if len(batch1["embed"]) < len(batch2["embed"]) else batch2
            size_difference = len(bigger_batch["embed"]) - len(smaller_batch["embed"])

            trimmed_batch = {"embed": bigger_batch["embed"][:-size_difference], "entity": bigger_batch["entity"][:-size_difference]}
            batch_reminder = {"embed": bigger_batch["embed"][-size_difference:], "entity": bigger_batch["entity"][-size_difference:]}
            self._compare_batches(trimmed_batch, smaller_batch)
            self._compare_asymmetric_batch(batch_reminder)

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
            try:
                files = os.listdir(checkpoint_path) if os.path.isdir(checkpoint_path) else [checkpoint_path]
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
                            best_threshold = self._get_best_threshold(fpr, tpr, thresholds, 0.05)

                            filepath = os.path.join(output_dir, "roc-" + str(checkpoint_number) + ".png")
                            if filepath:
                                self._save_roc(fpr, tpr, roc_auc, filepath)

                            stats = self._threshold_based_statistics(best_threshold)
                            self._print_pair_stats(stats, best_threshold, roc_auc)

                            processed_files.add(checkpoint_number)

                if len(processed_files) == prev_len:
                    break
            except Exception as e:
                print("Error occurred while processing files. \nError description: ", e)
                exit(-1)

    def _print_pair_stats(self, stats, threshold, roc_auc):
        print("-------- Model score ---------")
        print("AUC score: ", roc_auc)
        print("----- Precision accuracy -----")
        print("Selected threshold: ", threshold)
        print("F1 score: ", self._compute_f1(stats))
        print("----- General statistics -----")
        print("True positive: ", stats["true_positive"])
        print("False positive: ", stats["false_positive"])
        print("False negative: ", stats["false_negative"])
        print("True negative: ", stats["true_negative"])
        print("------------------------------")

    def _generate_roc(self):
        fpr, tpr, thresholds = roc_curve(self.labels, self.similarities)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, thresholds, roc_auc

    @staticmethod
    def _get_best_threshold(fpr, tpr, thresholds, max_t):
        indexes = [True if rate <= max_t else False for rate in fpr]
        best_index = numpy.argmax(tpr[indexes])
        return thresholds[indexes][best_index]

    @staticmethod
    def _save_roc(fpr, tpr, roc_auc, path):
        plt.figure()
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
        true_pos_norm = stats["true_positive"] / (stats["true_positive"] + stats["false_negative"])
        false_pos_norm = stats["false_positive"] / (stats["false_positive"] + stats["true_negative"])
        false_neg_norm = stats["false_negative"] / (stats["true_positive"] + stats["false_negative"])

        precision_divisor = (true_pos_norm + false_pos_norm)
        precision = true_pos_norm / precision_divisor if precision_divisor > 0 else 0
        recall_divisor = true_pos_norm + false_neg_norm
        recall = true_pos_norm / recall_divisor if recall_divisor > 0 else 0
        return 2 * ((precision * recall) / (precision + recall)) if precision + recall > 0 else 0

import numpy
import torch


class Metrics:
    def __init__(self, model, dataloader, config):
        self.model = model
        self.dataloader = dataloader
        self.config = config

    def pair_stats(self, accuracy):
        self.model.eval()

        l_embed = []
        l_entity = []
        stats = {
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
            "true_negative": 0,
            "len": 0
        }

        for idx, data_sample in enumerate(self.dataloader):
            image = data_sample["image"]
            entity = data_sample["entity"]

            with torch.no_grad():
                out_embeddings = self.model(image)

            if self.config.model_name == "dat":
                out_embeddings = out_embeddings[0]

            if idx % 2 == 0:
                l_embed = out_embeddings
                l_entity = entity

                if idx == len(self.dataloader) - 1:
                    self._compare_asymmetric_batch(
                        {"embed": l_embed.cpu(), "entity": l_entity.cpu()},
                        stats,
                        accuracy
                    )

            if idx % 2 == 1:
                self._compare_batches(
                    {"embed": l_embed.cpu(), "entity": l_entity.cpu()},
                    {"embed": out_embeddings.cpu(), "entity": entity.cpu()},
                    stats,
                    accuracy
                )

            # if idx == 50:
            #     break

        return stats

    def _compare_asymmetric_batch(self, batch, stats, accuracy):
        if len(batch["embed"]) % 2 != 0:
            batch = {"embed": batch["embed"][:-1], "entity": batch["entity"][:-1]}

        half_idx = len(batch["embed"]) // 2

        b1 = {"embed": batch["embed"][:half_idx], "entity": batch["entity"][:half_idx]}
        b2 = {"embed": batch["embed"][half_idx:], "entity": batch["entity"][half_idx:]}
        return self._compare_batches(b1, b2, stats, accuracy)

    def _compare_batches(self, batch1, batch2, stats, accuracy):
        if len(batch1["embed"]) == len(batch2["embed"]):
            for idx in range(len(batch1["embed"])):
                if self._similarity(batch1["embed"][idx], batch2["embed"][idx]) > accuracy:
                    if batch1["entity"][idx] == batch2["entity"][idx]:
                        print("Passed with similarity: ", self._similarity(batch1["embed"][idx], batch2["embed"][idx]))
                        stats["true_positive"] += 1
                    else:
                        print("**** Falsely passed with similarity: ", self._similarity(batch1["embed"][idx], batch2["embed"][idx]))
                        stats["false_positive"] += 1
                else:
                    if batch1["entity"][idx] == batch2["entity"][idx]:
                        print("Failed with similarity: ", self._similarity(batch1["embed"][idx], batch2["embed"][idx]))
                        stats["false_negative"] += 1
                    else:
                        stats["true_negative"] += 1
            stats["len"] = len(batch1)
        else:
            bigger_batch = batch1 if len(batch1["embed"]) > len(batch2["embed"]) else batch2
            smaller_batch = batch1 if len(batch1["embed"]) < len(batch2["embed"]) else batch2
            size_difference = len(bigger_batch["embed"]) - len(smaller_batch["embed"])

            trimmed_batch = {"embed": bigger_batch["embed"][:-size_difference], "entity": bigger_batch["entity"][:-size_difference]}
            batch_reminder = {"embed": bigger_batch["embed"][-size_difference:], "entity": bigger_batch["entity"][-size_difference:]}
            self._compare_batches(trimmed_batch, smaller_batch, stats, accuracy)
            self._compare_asymmetric_batch(batch_reminder, stats, accuracy)

    @staticmethod
    def _similarity(embedding1, embedding2):
        embedding1_magnitude = numpy.linalg.norm(embedding1)
        embedding2_magnitude = numpy.linalg.norm(embedding2)
        cosine_similarity = numpy.dot(embedding1, embedding2) / (embedding1_magnitude * embedding2_magnitude)
        return cosine_similarity

    @staticmethod
    def compute_f1(stats):
        precision_divisor = (stats["true_positive"] + stats["false_positive"])
        precision = stats["true_positive"] / precision_divisor if precision_divisor > 0 else 0
        recall_divisor = stats["true_positive"] + stats["false_negative"]
        recall = stats["true_positive"] / recall_divisor if recall_divisor > 0 else 0
        return 2 * ((precision * recall) / (precision + recall)) if precision + recall > 0 else 0

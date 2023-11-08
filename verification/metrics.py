import numpy
import torch

class Metrics:
    def __init__(self, model, dataloader, config):
        self.model = model
        self.dataloader = dataloader
        self.statistics = {}
        self.config = config

    def _update_statistics(self, entity, embedding):
        if entity not in self.statistics:
            self.statistics[entity] = [embedding]
        else:
            self.statistics[entity].append(embedding)

    def _run_statistics(self):
        self.model.eval()
        counter = 0

        seen_classes = []
        for data_sample in self.dataloader:
            counter += 1

            image = data_sample["image"]
            entity = data_sample["entity"]

            with torch.no_grad():
                out_embeddings = self.model(image)

            if self.config.model_name == "dat":
                out_embeddings = out_embeddings[0]

            #out_embeddings = torch.nn.functional.normalize(out_embeddings, dim=1)

            for idx, embedding in enumerate(out_embeddings):
                if entity[idx].item() not in seen_classes:
                    print(embedding)
                    seen_classes.append(entity[idx].item())
                self._update_statistics(entity[idx].item(), embedding.cpu().numpy())

            if counter > 200:
              return

    def _cluster_center(self, class_id):
        matrix = numpy.asmatrix(self.statistics[class_id])
        #print("Max matrix value: ", matrix.max())
        argmax = matrix.argmax()
        y = argmax // matrix.shape[1]
        x = argmax % matrix.shape[1]

        #print("Where it is: x", x, "; y", y)
        #print(self.statistics[class_id])
        #print("Shape: ", matrix.shape)
        #print("Correction: ", matrix[y][x])
        return numpy.mean(matrix, axis=0).tolist()[0]


    def _similarity(self, embedding1, embedding2):
        embedding1_magnitude = numpy.linalg.norm(embedding1)
        embedding2_magnitude = numpy.linalg.norm(embedding2)
        cosine_similarity = numpy.dot(embedding1, embedding2) / (embedding1_magnitude * embedding2_magnitude)
        return cosine_similarity

    def _flatten(self, l):
        return [item for sublist in l for item in sublist]

    def _test_metrics(self):
        for class_label_a in self.statistics:
            for class_label_b in self.statistics:
                sum = 0

                for embedding_a in self.statistics[class_label_a]:
                    for embedding_b in self.statistics[class_label_b]:
                        embedding_magnitude_a = numpy.linalg.norm(embedding_a)
                        embedding_magnitude_b = numpy.linalg.norm(embedding_b)
                        cosine_similarity = numpy.dot(embedding_a, embedding_b) / (
                                embedding_magnitude_a * embedding_magnitude_b)
                        sum += cosine_similarity

                print("Class ", class_label_a, " and class ", class_label_b, " have similarity: ", sum / (len(self.statistics[class_label_a]) * len(self.statistics[class_label_b])))




        # with torch.no_grad():
        #     avg = 0
        #
        #     weights = self.statistics
        #     for i in range(weights.shape[0]):
        #         embedding1 = weights[i]
        #
        #         num_neg_class = 0
        #         num_pos_class = 0
        #         num_neg = 0
        #         num_pos = 0
        #         for j in range(weights.shape[0]):
        #             embedding2 = weights[j]
        #
        #             embedding1_magnitude = torch.linalg.norm(embedding1)
        #             embedding2_magnitude = torch.linalg.norm(embedding2)
        #             cosine_similarity = torch.dot(embedding1, embedding2) / (
        #                         embedding1_magnitude * embedding2_magnitude)
        #
        #             if cosine_similarity < 0:
        #                 if entity[i] == entity[j]:
        #                     num_neg_class += 1
        #                 num_neg += 1
        #             else:
        #                 if entity[i] == entity[j]:
        #                     num_pos_class += 1
        #                 num_pos += 1
        #             avg += cosine_similarity
        #
        #         print("------------------")
        #         print("Class id: ", entity[i].item())
        #         print("SUM: ", avg)
        #         print("Len: ", weights.shape)
        #         print("Neg: ", num_neg)
        #         print("Pos: ", num_pos)
        #         print("Neg class: ", num_neg_class)
        #         print("Pos class: ", num_pos_class)
        #
        #         print("Running avg: ", avg / (i * weights.shape[0]))
        #     print("Average distance: ", avg / (weights.shape[0] * weights.shape[0]))
        # return

    def _cluster_distance(self, class_id, embedding):
        center = numpy.array(self._cluster_center(class_id))

        center_magnitude = numpy.linalg.norm(center)
        embedding_magnitude = numpy.linalg.norm(embedding)
        cosine_similarity = numpy.dot(center, embedding) / (center_magnitude * embedding_magnitude)
        # normalized_center = center / numpy.linalg.norm(center)
        # normalized_embedding = embedding / numpy.linalg.norm(embedding)
        # print("--")
        # print("Normalized distance: ", numpy.linalg.norm(normalized_center - normalized_embedding))
        # print("Non-normalized distance: ", numpy.linalg.norm(class_id - embedding))
        # print("Similarity: ", numpy.dot(normalized_center, normalized_embedding))
        # print("--")
        return cosine_similarity

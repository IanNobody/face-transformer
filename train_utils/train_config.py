# Author: Šimon Strýček
# Description: Configuration for training/testing scripts.
# Year: 2024

import copy
import itertools


def build_config(args):
    return TrainingConfiguration(
        model_name=args.model,
        dataset_name=args.dataset,
        devices=args.gpu,
        checkpoint_count=args.checkpoint_count,
        checkpoints_dir=args.checkpoints_dir,
        weights_file_path=args.weights_file_path,
        batch_size=args.batch_size,
        num_of_epoch=args.num_epoch,
        old_checkpoint_format=args.old_format
    )

class TrainingConfiguration:
    def __init__(self,
                 model_name,
                 dataset_name,
                 devices=[],
                 checkpoint_count=10,
                 checkpoints_dir='./',
                 weights_file_path='./weight.pth',
                 batch_size=32,
                 num_of_epoch=50,
                 min_model_lr=1e-6,
                 max_model_lr=1e-7,
                 min_crit_lr=1e-3,
                 max_crit_lr=1e-2,
                 embedding_size=512,
                 embedding_loss_rate=0.85,
                 warmup_epochs=10,
                 old_checkpoint_format=False):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.devices = devices
        self.data_sampler_state_file = None
        self.checkpoint_count = checkpoint_count
        self.checkpoints_dir = checkpoints_dir
        self.weights_file_path = weights_file_path
        self.batch_size = batch_size
        self.num_of_epoch = num_of_epoch
        self.min_model_lr = min_model_lr
        self.max_model_lr = max_model_lr
        self.min_crit_lr = min_crit_lr
        self.max_crit_lr = max_crit_lr
        self.embedding_size = embedding_size
        self.embedding_loss_rate = embedding_loss_rate
        self.warmup_epochs = warmup_epochs
        self.old_checkpoint_format = old_checkpoint_format

    def sane_lr_values(self):
        return [1e-4, 1e-2, 1e-1]

    def sane_embedding_sizes(self):
        return [512]

    def sane_embedding_loss_rates(self):
        return [0.6]

    def generate_all_permutations(self):
        permutations = list(itertools.product(
            [1e-6],
            self.sane_lr_values(),
            self.sane_embedding_sizes(),
            self.sane_embedding_loss_rates()
        ))
        for max_model_lr, max_crit_lr, emb_size, emb_loss_rate in permutations:
            self_copy = copy.deepcopy(self)
            self_copy.max_model_lr = max_model_lr
            self_copy.min_model_lr = max_crit_lr / 10
            self_copy.max_crit_lr = max_crit_lr
            self_copy.min_crit_lr = max_crit_lr / 10
            self_copy.embedding_size = emb_size
            self_copy.embedding_loss_rate = emb_loss_rate
            yield self_copy




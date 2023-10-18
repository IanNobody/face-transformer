import random
import torch


class TrainingConfiguration():
    def __init__(self,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 sampler_file_path=None,
                 sampler_seed=random.randint(1, 255),
                 checkpoint_count=5,
                 sample_freq=150,
                 export_weights_dir='./',
                 import_weight_path='./weight.pth',
                 batch_size=32,
                 num_of_epoch=50):
        self.device = device
        self.data_sampler_state_file = None
        self.sampler_file_path = sampler_file_path
        self.sampler_seed = sampler_seed
        self.checkpoint_count = checkpoint_count
        self.sample_freq = sample_freq
        self.export_weights_dir = export_weights_dir
        self.import_weight_path = import_weight_path
        self.batch_size = batch_size
        self.num_of_epoch = num_of_epoch
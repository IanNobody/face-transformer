import random
import torch


class TrainingConfiguration():
    def __init__(self,
                 model_name,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 sampler_seed=random.randint(1, 255),
                 checkpoint_count=5,
                 checkpoint_freq=150,
                 export_weights_dir='./',
                 checkpoint_path='./weight.pth',
                 batch_size=32,
                 num_of_epoch=50):
        self.model_name = model_name
        self.device = device
        self.data_sampler_state_file = None
        self.sampler_seed = sampler_seed
        self.checkpoint_count = checkpoint_count
        self.checkpoint_freq = checkpoint_freq
        self.export_weights_dir = export_weights_dir
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.num_of_epoch = num_of_epoch

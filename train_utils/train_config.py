def build_config(args):
    return TrainingConfiguration(
        model_name=args.model,
        devices=args.gpu,
        checkpoint_count=args.checkpoint_count,
        checkpoints_dir=args.checkpoints_dir,
        weights_file_path=args.weights_file_path,
        batch_size=args.batch_size,
        num_of_epoch=args.num_epoch
    )

class TrainingConfiguration:
    def __init__(self,
                 model_name,
                 devices=[],
                 checkpoint_count=10,
                 checkpoints_dir='./',
                 weights_file_path='./weight.pth',
                 batch_size=32,
                 num_of_epoch=50,
                 min_model_lr=5e-8,
                 max_model_lr=1e-6,
                 min_crit_lr=1e-6,
                 max_crit_lr=1e-4,
                 embedding_size=512,
                 warmup_epochs=10):
        self.model_name = model_name
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
        self.warmup_epochs = warmup_epochs

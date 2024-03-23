import argparse
import torch

import os
import glob

from data.datasets.lfw_data import LFWDataset
from models.lightning_wrapper import LightningWrapper
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader, ConcatDataset
from pytorch_metric_learning import losses
import torchvision.transforms as T
import albumentations as alb

from data.dataset_factory import build_train_dataset, build_eval_dataset
from models.model_factory import build_model


from train_utils.train_config import build_config
from verification.metrics import Metrics
from pytorch_lightning.loggers import TensorBoardLogger
import gc
import wandb
from pytorch_lightning.loggers import WandbLogger

torch.set_float32_matmul_precision('medium')

wandb.init(
    project="face_transformer",
    config={
        "architecture": "OpenCLIP",
        "dataset": "MS1Mv3",
    }
)

def start_training(model, dataloader, val_dataloader, config, classes):
    if config.weights_file_path is not None:
        lightning_model = LightningWrapper.load_from_checkpoint(config.weights_file_path, model=model, config=config,
                                                                num_classes=classes,
                                                                num_batches=len(dataloader) / len(config.devices),
                                                                map_location="cpu")
    else:
        lightning_model = LightningWrapper(model=model, config=config, num_classes=classes,
                                           num_batches=len(dataloader) / len(config.devices))
    checkpointer = ModelCheckpoint(
        dirpath=config.checkpoints_dir,
        filename='checkpoint-{epoch:02d}-{loss:.2f}-{acc:.2f}'
    )
    wandb_logger = WandbLogger(project='face_transformer')
    trainer = Trainer(max_epochs=config.num_of_epoch,
                      logger=wandb_logger,
                      overfit_batches=30,
                      callbacks=[checkpointer],
                      strategy='ddp_find_unused_parameters_true',
                      accelerator="auto", devices=config.devices)
    trainer.fit(lightning_model, dataloader, val_dataloader)
    print("Training successfully finished.")

def print_config_sumup(config, args, dataset):
    print("--------------------------------------")
    print("Training hyperparameters:")
    print("--------------------------------------")
    print("Model                |   " + args.model)
    print("Number of epochs     |   " + str(config.num_of_epoch))
    print("Batch size           |   " + str(config.batch_size))
    print("Active checkpoints   |   " + str(config.checkpoint_count))
    print("--------------------------------------")
    print("Dataset information:")
    print("--------------------------------------")
    print("Dataset name         |   " + args.dataset)
    print("Number of images     |   " + str(len(dataset)))
    print("Number of classes    |   " + str(dataset.num_classes()))
    print("--------------------------------------")

def augumentations():
    return alb.Compose([
        alb.RandomFog(p=0.35),
        alb.RandomBrightnessContrast(p=0.5),
        alb.Sharpen(p=0.4),
        alb.RGBShift(p=0.5),
        alb.AdvancedBlur(p=0.35),
        alb.HorizontalFlip(p=0.3),
        alb.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.4)
    ])

def create_argument_parser():
    parser = argparse.ArgumentParser(
        description="A script for testing face recognition models on a dataset of images"
    )

    parser.add_argument("--model", type=str, help="Model selection")
    parser.add_argument("--dataset", type=str, help="Dataset selection")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model on the dataset")
    parser.add_argument("--data_path", type=str, help="Path to the dataset directory")
    parser.add_argument("--annotation_dir", type=str, help="Path to the files list")
    parser.add_argument("--checkpoint_count", type=int, help="Number of checkpoints to keep", default=50)
    parser.add_argument("--checkpoints_dir", type=str, help="Path where to save checkpoints", default="./")
    parser.add_argument("--weights_file_path", type=str, help="Path to weights file", default=None)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size", default=32)
    parser.add_argument("-e", "--num_epoch", type=int, help="Number of epochs", default=50)
    parser.add_argument("--gpu", nargs='+', type=int, help="Which GPU unit to use (default is 0)", default=0)
    parser.add_argument("--output_dir", type=str, help="Path where to save evaluation stats", default=None)

    return parser

if __name__ == '__main__':
    parser = create_argument_parser()
    args = parser.parse_args()
    configuration = build_config(args)

    train_dataset = build_train_dataset(args.dataset, args.data_path, args.annotation_dir, augumentations())
    eval_dataset = build_eval_dataset(None)
    eval_dataloader = DataLoader(eval_dataset, batch_size=configuration.batch_size, shuffle=False,
                                 num_workers=8, collate_fn=LFWDataset.collate_fn)

    number_of_classes = train_dataset.num_classes()
    model = build_model(args.model, 512, number_of_classes)

    if not args.eval:
        train_dataloader = DataLoader(train_dataset, batch_size=configuration.batch_size, shuffle=True, num_workers=6)
        print_config_sumup(configuration, args, train_dataset)
        start_training(model, train_dataloader, eval_dataloader, configuration, number_of_classes)
    # else:
        # search_pattern = os.path.join(configuration.checkpoint_path, '**', '*.ckpt')
    #     for idx, ckpt_file in enumerate(sorted(glob.glob(search_pattern, recursive=True))):
    #         configuration.device = torch.device("cuda:"+str(args.gpu[0]))
    #         # model = MultitaskOpenCLIP(None, 8631)
    #         model = OpenCLIPWrapper(100000)
    #         # crit = losses.ArcFaceLoss(8631, embedding_size)
    #         crit = losses.ArcFaceLoss(100000, 512)
    #         model = LightningWrapper.load_from_checkpoint(ckpt_file, model=model, config=configuration, criterion=crit, map_location=configuration.device)
    #         model.eval()
    #         metrics = Metrics(model, dataloader, configuration)
    #         print("Checking file: ", ckpt_file)
    #         metrics.test_and_print(args.output_dir, idx)
    #         del model
    #         del metrics
    #         gc.collect()
    #         torch.cuda.empty_cache()


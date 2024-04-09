import argparse
import gc
import torch
import os
import glob

from data.datasets.lfw_data import LFWDataset
from models.lightning_wrapper import LightningWrapper
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader
import albumentations as alb

from data.dataset_factory import build_train_dataset, build_eval_dataset
from models.model_factory import build_model
from train_utils.train_config import build_config
import wandb
from pytorch_lightning.loggers import WandbLogger

from verification.metrics import test_and_print

torch.set_float32_matmul_precision('medium')

def init_training_wandb(config, project_name):
    wandb.init(
        project=project_name,
        config={
            "architecture": config.model_name,
            "dataset": config.dataset_name,
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
        save_top_k=-1,
        filename='checkpoint-{epoch:02d}-{loss:.2f}-{acc:.2f}'
    )
    wandb_logger = WandbLogger(project='face_transformer')
    trainer = Trainer(max_epochs=config.num_of_epoch,
                      logger=wandb_logger,
                      callbacks=[checkpointer],
                      #overfit_batches=200,
                      #strategy='ddp_find_unused_parameters_true',
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
    parser.add_argument("--hyper", action="store_true", help="Starts hyperparameters testing")
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
    model = build_model(configuration, 512, number_of_classes)

    if not args.eval:
        if not args.hyper:
            init_training_wandb(configuration, "face_transformer_cosface_adamw")
            train_dataloader = DataLoader(train_dataset, batch_size=configuration.batch_size, shuffle=True, num_workers=6)
            print_config_sumup(configuration, args, train_dataset)
            start_training(model, train_dataloader, eval_dataloader, configuration, number_of_classes)
        else:
            for config in configuration.generate_all_permutations():
                wandb.init(
                    reinit=True,
                    project="face_transformer_hyperparams_study-cosface",
                    name="e" + str(config.embedding_size) + "_r" + str(config.embedding_loss_rate)
                         + "_m" + str(config.max_model_lr) + "_c" + str(config.max_crit_lr),
                    config={
                        "architecture": "OpenCLIP",
                        "dataset": "MS1Mv3",
                        "min_model_lr": config.min_model_lr,
                        "max_model_lr": config.max_model_lr,
                        "min_crit_lr": config.min_crit_lr,
                        "max_crit_lr": config.max_crit_lr,
                        "embedding_size": config.embedding_size,
                        "embedding_loss_rate": config.embedding_loss_rate
                    }
                )
                model = build_model(config, config.embedding_size, number_of_classes)
                train_dataloader = DataLoader(train_dataset, batch_size=configuration.batch_size, shuffle=True, num_workers=6)
                start_training(model, train_dataloader, eval_dataloader, config, number_of_classes)
                wandb.finish()
    else:
        search_pattern = os.path.join(configuration.weights_file_path, '**', '*.ckpt')
        for idx, ckpt_file in enumerate(sorted(glob.glob(search_pattern, recursive=True))):
            configuration.device = torch.device("cuda:"+str(args.gpu[0]))
            lightning_model = LightningWrapper.load_from_checkpoint(
                ckpt_file, model=model,
                config=configuration,
                num_classes=number_of_classes,
                map_location=configuration.device
            )

            print("Checking file: ", ckpt_file)
            test_and_print(model, eval_dataloader, configuration.device, args.output_dir, idx)
            del lightning_model
            gc.collect()
            torch.cuda.empty_cache()


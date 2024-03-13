import argparse
import torch

from models.DAT.dat import DAT
from models.Flatten_T.flatten_swin import FLattenSwinTransformer
from models.SMT.smt import SMT
from models.BiFormer.biformer import biformer_base
from models.CMT.cmt import cmt_b
from models.NoisyViT.noisy_vit import vit_b
from models.OpenCLIP.pretrained_openclip import OpenCLIPWrapper
from models.OpenCLIP.multitask_openclip import MultitaskOpenCLIP
from models.LightningModule import LightningWrapper
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torchvision.models as models

from torch.utils.data import DataLoader, ConcatDataset
from pytorch_metric_learning import losses
import torch.nn as nn
import torchvision.transforms as T
import albumentations.augmentations.transforms as A
import albumentations as alb

from data.celeba_data import CelebADataset
from data.vggface_data import VGGFaceDataset
from data.lfw_data import LFWDataset
from train_utils.resumable_sampler import ResumableRandomSampler
from train_utils.train_config import TrainingConfiguration
from verification.metrics import Metrics
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision('medium')

warmup_epochs = 1
max_model_lr = 1e-5
min_model_lr = 5e-7
max_crit_lr = 4e-5
min_crit_lr = 1e-6
embedding_size = 512


def start_training(model, dataloader, val_dataloader, config, classes):
    criterion = losses.ArcFaceLoss(num_of_classes, embedding_size)
    lightning_model = LightningWrapper(model, config, max_model_lr, min_model_lr, criterion, max_crit_lr, min_crit_lr,
                                       warmup_epochs, len(dataloader) / len(config.device))
    checkpointer = ModelCheckpoint(
        dirpath=config.export_weights_dir,
        filename='checkpoint-{epoch:02d}-{loss:.2f}-{acc:.2f}',
        monitor='acc',
        save_top_k=10,
        mode='max'
    )
    logger = TensorBoardLogger(save_dir="logs/", name="cmt_refactored")
    trainer = Trainer(max_epochs=config.num_of_epoch,
                      logger=logger,
                      callbacks=[checkpointer],
                      strategy='ddp_find_unused_parameters_true',
                      accelerator="auto", devices=config.device)
    trainer.fit(lightning_model, dataloader, val_dataloader)
    print("Training successfully finished.")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_config_sumup(config, dataset, model, num_of_classes):
    print("*****************************************************")
    print("Model details: ")
    print("- Model name:" + str(config.model_name))
    print("- Number of parameters:" + str(count_parameters(model)))
    print("*****************************************************")
    print("Beginning training with the following parameters:")
    print("- Number of epochs:" + str(config.num_of_epoch))
    print("- Minibatch size:" + str(config.batch_size))
    print("- Number of active checkpoints:" + str(config.checkpoint_count))
    print("- Checkpoint frequency:" + str(config.checkpoint_freq))
    print("*****************************************************")
    print("Dataset information:")
    print("- Dataset path:" + str(args.dataset_path))
    print("- Files catalog:" + str(args.files_list))
    print("- Number of classes:" + str(num_of_classes))
    print("- Number of samples:" + str(len(dataset)))
    print("*****************************************************")


def transforms():
    return T.Compose([
        T.ToTensor(),
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        # T.Normalize([0.48579846, 0.4084752675, 0.3738937391],
        #             [0.2683508996, 0.2441760085, 0.2375956649]),
        T.Lambda(lambda x: torch.clamp(x, 0, 1)),
        lambda x: x.to(torch.float32)
    ])


def augumentations():
    return alb.Compose([
        # A.RandomFog(p=0.3),
        # A.Equalize(mode="cv", by_channels=True, p=0.3),
        # A.RandomBrightnessContrast(p=0.3),
        # A.Sharpen(p=0.3),
        A.Normalize()
    ])


def dataset(args, transform, augmentation):
    datasets = []

    if args.vggface:
        datasets.append(VGGFaceDataset(
            args.dataset_path,
            args.files_list,
            transform=transform,
            augmentation=augmentation
        ))
    elif args.celeba:
        datasets.append(CelebADataset(
            args.annotation_path,
            args.dataset_path,
            transform=transform,
            augumentation=augmentation
        ))
    elif args.lfw:
        datasets.append(LFWDataset(transform=transform))

    return ConcatDataset(datasets), sum([d.num_of_classes() for d in datasets])


def create_model(args, configuration, embedding_size, num_of_classes):
    model = None
    if args.swin:
        configuration.model_name = "swin"
        model = models.swin_t()
        model.head = nn.Linear(model.head.in_features, embedding_size)
    elif args.resnet_50:
        configuration.model_name = "resnet_50"
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, embedding_size)
    elif args.dat:
        configuration.model_name = "dat"
        model = DAT(num_classes=num_of_classes)
        model.cls_head = nn.Linear(model.cls_head.in_features, embedding_size)
    elif args.flatten_transformer:
        configuration.model_name = "flatten_transformer"
        model = FLattenSwinTransformer(num_classes=num_of_classes)
        model.head = nn.Linear(model.head.in_features, embedding_size)
    elif args.smt:
        configuration.model_name = "smt"
        model = SMT(num_classes=num_of_classes)
        model.embed_fc = nn.Linear(model.embed_fc.in_features, embedding_size)
    elif args.biformer:
        configuration.model_name = "biformer"
        model = biformer_base()
        model.head = nn.Linear(model.head.in_features, embedding_size)
    elif args.cmt:
        configuration.model_name = "cmt"
        model = cmt_b(num_classes=num_of_classes)
        model.embed_fc = nn.Linear(model.head.in_features, embedding_size)
    elif args.noisy_vit:
        configuration.model_name = "noisy_vit"
        model = vit_b()
        model.head = nn.Linear(model.head.in_features, embedding_size)
    elif args.openclip:
        configuration.model_name = "openclip"
        model = OpenCLIPWrapper()
    elif args.multitask_openclip:
        configuration.model_name = "multitask_openclip"
        model = MultitaskOpenCLIP(configuration.device, num_of_classes)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="A script for testing face recognition models on a dataset of images"
    )

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--swin", action="store_true", help="Use Swin Transformer model")
    model_group.add_argument("--resnet_50", action="store_true", help="Use ResNet-50 model")
    model_group.add_argument("--dat", action="store_true", help="Use dynamic attention model")
    model_group.add_argument("--flatten_transformer", action="store_true", help="Use flatten transformer model")
    model_group.add_argument("--smt", action="store_true", help="Scale aware modulation transformer")
    model_group.add_argument("--biformer", action="store_true", help="Use BiFormer model")
    model_group.add_argument("--cmt", action="store_true", help="Use CMT ViT-CNN hybrid model")
    model_group.add_argument("--noisy_vit", action="store_true", help="Use Noisy ViT model")
    model_group.add_argument("--openclip", action="store_true", help="Use OpenAI CLIP model")
    model_group.add_argument("--multitask_openclip", action="store_true",
                             help="Use OpenAI CLIP model with multitask learning")

    dataset_group = parser.add_argument_group()
    dataset_group.add_argument("--celeba", action="store_true", help="Use CelebA dataset")
    dataset_group.add_argument("--vggface", action="store_true", help="Use VGGFace2 dataset")
    dataset_group.add_argument("--cacd", action="store_true", help="Use cross-age CACD dataset")
    dataset_group.add_argument("--lfw", action="store_true", help="Use LFW benchmark dataset")

    parser.add_argument("--eval", action="store_true", help="Evaluate the model on the dataset")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset directory")
    parser.add_argument("--files_list", type=str, help="Path to the files list")
    parser.add_argument("--checkpoint_count", type=int, help="Number of checkpoints to keep", default=5)
    parser.add_argument("--checkpoint_freq", type=int, help="Frequency of checkpoints", default=150)
    parser.add_argument("--checkpoints_dir", type=str, help="Path where to save checkpoints", default="./")
    parser.add_argument("-b", "--batch_size", type=int, help="Minibatch size", default=32)
    parser.add_argument("-e", "--num_of_epoch", type=int, help="Number of epochs", default=50)
    parser.add_argument("--checkpoint_path", type=str,
                        help="Path to the checkpoint file or directory of checkpoints", default=None)
    parser.add_argument("--output_dir", type=str, help="Path where to store model statistics", default=None)
    parser.add_argument("--gpu", nargs='+', type=int, help="Which GPU unit to use (default is 0)", default=0)

    args = parser.parse_args()

    configuration = TrainingConfiguration(
        model_name=None,
        device=args.gpu,
        checkpoint_count=args.checkpoint_count,
        checkpoint_freq=args.checkpoint_freq,
        export_weights_dir=args.checkpoints_dir,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        num_of_epoch=args.num_of_epoch
    )

    dataset, num_of_classes = dataset(args, transforms(), augumentations())
    model = create_model(args, configuration, embedding_size, num_of_classes)

    if not args.eval:
        data_sampler = ResumableRandomSampler(dataset, configuration)
        dataloader = DataLoader(dataset, batch_size=configuration.batch_size, sampler=data_sampler, num_workers=5)
        val_data = LFWDataset(transform=transforms())
        val_dataloader = DataLoader(val_data, batch_size=configuration.batch_size,
                                    num_workers=8, collate_fn=LFWDataset.collate_fn, shuffle=True)
        print_config_sumup(configuration, dataset, model, num_of_classes)
        start_training(model, dataloader, val_dataloader, configuration, num_of_classes)
    else:
        print("Selected device - cuda:", str(args.gpu[0]))
        configuration.device = torch.device("cuda:"+str(args.gpu[0]))
        model = MultitaskOpenCLIP(None, 8631)
        crit = losses.ArcFaceLoss(8631, embedding_size)
        model = LightningWrapper.load_from_checkpoint(configuration.checkpoint_path, model=model, config=configuration, criterion=crit)
        dataloader = DataLoader(dataset, batch_size=configuration.batch_size, num_workers=16,
                                shuffle=True, collate_fn=LFWDataset.collate_fn)
        metrics = Metrics(model, dataloader, configuration)
        metrics.test_and_print(args.output_dir)

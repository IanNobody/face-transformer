import argparse
import torchvision.models as models
from torch import optim
from torch.utils.data import DataLoader
from pytorch_metric_learning import losses

from data.celeba_data import CelebADataset
from training.resumable_sampler import ResumableRandomSampler
from training.train import train
from training.train_config import TrainingConfiguration


def download(fileUrl):
    raise NotImplementedError("Download functionality is not yet implemented.")


def start_training(model, dataset, config):
    data_sampler = ResumableRandomSampler(dataset, config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=data_sampler)
    optimizer = optim.Adam(model.parameters())
    classes = dataset.num_of_classes()
    criterion = losses.ArcFaceLoss(classes + 1, classes)
    train(model, dataloader, optimizer, criterion, config.device, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="A script for testing face recognition models on a dataset of images"
    )

    browser_group = parser.add_mutually_exclusive_group(required=True)
    browser_group.add_argument("--swin", action="store_true", help="Use Swin Transformer model")
    browser_group.add_argument("--cmt", action="store_true", help="Use CMT ViT-CNN hybrid model")
    browser_group.add_argument("--dconv", action="store_true", help="Use a CNN model based on dynamic convolutions")

    dataset_group = parser.add_mutually_exclusive_group(required=True)
    browser_group.add_argument("--celeba", action="store_true", help="Use CelebA dataset")
    browser_group.add_argument("--megaface", action="store_true", help="Use MegaFace dataset")

    parser.add_argument("--dataset-path", type=str, help="Path to the dataset directory", required=True)
    parser.add_argument("-d", "--download", action="store_true", help="Download the dataset if it is not present")
    parser.add_argument("--checkpoint_count", type=int, help="Number of checkpoints to keep", default=5)
    parser.add_argument("--sample_freq", type=int, help="Frequency of checkpoints", default=150)
    parser.add_argument("-b", "--batch_size", type=int, help="Minibatch size", default=32)
    parser.add_argument("-e", "--num_of_epoch", type=int, help="Number of epochs", default=50)
    parser.add_argument("--checkpoint_path", type=str, help="Path where to save checkpoints", default="./")
    parser.add_argument("--weight", type=str, help="Path to a model weights file")
    parser.add_argument("--sampler", type=str, help="Path to a sampler state file")

    args = parser.parse_args()
    configuration = TrainingConfiguration(
        sampler_file_path=args.weight,
        checkpoint_count=args.checkpoint_count,
        sample_freq=args.sample_freq,
        export_weights_dir=args.checkpoint_path,
        import_weight_path=args.sampler,
        batch_size=args.batch_size,
        num_of_epoch=args.num_of_epoch
    )

    dataset_path = args.dataset_path
    dataset = None
    if args.download:
        dataset_url = "celeba" if args.celeba else "megaface"
        dataset_path = download(dataset_url)

    if args.megaface:
        raise NotImplementedError("MegaFace dataset is not yet supported.")
    elif args.celeba:
        raise NotImplementedError("CelebA dataset is not yet supported.")

    model = None
    if args.swin:
        model = models.swin_t()
    elif args.cmt:
        raise NotImplementedError("CMT model is not yet supported.")
    elif args.dconv:
        raise NotImplementedError("D-CNN model is not yet supported.")

    start_training(model, dataset, configuration)
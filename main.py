import argparse

import torch

from models.DAT.dat import DAT
from models.Flatten_T.flatten_swin import FLattenSwinTransformer
from models.SMT.smt import SMT
import torchvision.models as models
from torch import optim
from torch import device
from torch import cuda
from torch.utils.data import DataLoader
from pytorch_metric_learning import losses
import torch.nn as nn
import torchvision.transforms as T
from safe_gpu import safe_gpu

from data.celeba_data import CelebADataset
from data.vggface_data import VGGFaceDataset
from training.resumable_sampler import ResumableRandomSampler
from training.train import train
from training.train_config import TrainingConfiguration
from verification.metrics import Metrics


def start_training(model, dataset, config):
    data_sampler = ResumableRandomSampler(dataset, config)
    resume_sampler(data_sampler, config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=data_sampler)
    classes = dataset.num_of_classes()
    criterion = losses.ArcFaceLoss(classes, 512).to(config.device)
    model_optimizer = optim.Adam(model.parameters(), lr=0.00004)
    loss_optimizer = optim.SGD(criterion.parameters(), lr=0.01)
    train(model, dataloader, model_optimizer, loss_optimizer, criterion, config.device, config)
    print("Training succesfully finished.")


def resume_sampler(sampler, config):
    if config.sampler_file_path:
        sampler_state = torch.load(config.sampler_file_path)
        sampler.set_state(sampler_state)


def print_config_sumup(config, dataset):
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
    print("- Number of classes:" + str(dataset.num_of_classes()))
    print("- Number of samples:" + str(len(dataset)))
    print("*****************************************************")


def transforms():
    return T.Compose([
        T.ToTensor(),
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.Normalize([0.48579846, 0.4084752675, 0.3738937391],
                    [0.2683508996, 0.2441760085, 0.2375956649]),
        lambda x: x.to(torch.float32)
    ])

def dataset(args, device, transform):
    if args.vggface:
        return VGGFaceDataset(
            args.dataset_path,
            args.files_list,
            device=device,
            transform=transform
        )
    elif args.celeba:
        return CelebADataset(
            args.annotation_path,
            args.dataset_path,
            device=device,
            transform=transform
        )


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
    # model_group.add_argument("--cmt", action="store_true", help="Use CMT ViT-CNN hybrid model")

    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--celeba", action="store_true", help="Use CelebA dataset")
    dataset_group.add_argument("--vggface", action="store_true", help="Use VGGFace2 dataset")
    #dataset_group.add_argument("--ms-celeb", action="store_true", help="Use MS-Celeb-1M dataset")

    parser.add_argument("--eval", action="store_true", help="Evaluate the model on the dataset")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset directory", required=True)
    # parser.add_argument("--annotation_path", type=str, help="Path to the annotation file", required=True)
    parser.add_argument("--files_list", type=str, help="Path to the files list", required=True)
    parser.add_argument("--checkpoint_count", type=int, help="Number of checkpoints to keep", default=5)
    parser.add_argument("--checkpoint_freq", type=int, help="Frequency of checkpoints", default=150)
    parser.add_argument("--checkpoint_path", type=str, help="Path where to save checkpoints", default="./")
    parser.add_argument("-b", "--batch_size", type=int, help="Minibatch size", default=32)
    parser.add_argument("-e", "--num_of_epoch", type=int, help="Number of epochs", default=50)
    parser.add_argument("--weight", type=str, help="Path to a model weights file")
    parser.add_argument("--sampler", type=str, help="Path to a sampler state file")
    parser.add_argument("--gpu", type=int, help="Which GPU unit to use (default is 0)", default=0)

    args = parser.parse_args()
    #safe_gpu.claim_gpus()

    configuration = TrainingConfiguration(
        model_name=None,
        device=device("cuda:" + str(args.gpu) if cuda.is_available() else "cpu"),
        sampler_file_path=args.sampler,
        checkpoint_count=args.checkpoint_count,
        checkpoint_freq=args.checkpoint_freq,
        export_weights_dir=args.checkpoint_path,
        import_weight_path=args.weight,
        batch_size=args.batch_size,
        num_of_epoch=args.num_of_epoch
    )

    dataset = dataset(args, configuration.device, transforms())

    if args.swin:
        configuration.model_name = "swin"
        model = models.swin_t()
        model.head = nn.Linear(model.head.in_features, 512)
    elif args.resnet_50:
        configuration.model_name = "resnet_50"
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, 512)
    elif args.dat:
        configuration.model_name = "dat"
        model = DAT(num_classes=dataset.num_of_classes())
        model.cls_head = nn.Linear(model.cls_head.in_features, 512)
    elif args.flatten_transformer:
        configuration.model_name = "flatten_transformer"
        model = FLattenSwinTransformer(num_classes=dataset.num_of_classes())
        model.head = nn.Linear(model.head.in_features, 512)
    else:
        configuration.model_name = "smt"
        model = SMT(num_classes=dataset.num_of_classes())
        model.head = nn.Linear(model.head.in_features, 512)

    model = model.to(configuration.device)

    if args.weight:
        print("Loading weights from ", args.weight)
        state_dict = torch.load(args.weight, map_location=configuration.device)
        model.load_state_dict(state_dict)

    if not args.eval:
        print_config_sumup(configuration, dataset)
        start_training(model, dataset, configuration)
    else:
        dataloader = DataLoader(dataset, batch_size=configuration.batch_size)
        metrics = Metrics(model, dataloader, configuration)
        metrics._run_statistics()
        #metrics._test_metrics()

        for sample in metrics.statistics.keys():
            item = metrics.statistics[sample][0]

            score = 0
            matching_class = None

            print("*****************************************************")
            print("Sample: ", sample)

            class_center = metrics._cluster_center(sample)

            for entity in metrics.statistics.keys():
                item2 = metrics.statistics[entity][0]
                dist = metrics._cluster_distance(entity, item)
                target_center = metrics._cluster_center(entity)
                dist2 = metrics._similarity(item, item2)

                print("---------------")
                print(entity, "> ", metrics._similarity(class_center, target_center))
                print("> sample-cluster: ", dist)
                print("> sample-sample: ", dist2)
                print("---------------")

                if dist > score:
                    score = dist
                    matching_class = entity

            print("*****************************************************")
            print("Sample of class ", sample, " is closes to ", matching_class, " class center.")
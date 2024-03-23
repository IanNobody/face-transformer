from data.datasets.celeba_data import CelebADataset
from data.datasets.vggface_data import VGGFaceDataset
from data.datasets.ms1m_dataset import MS1M_Dataset
from data.datasets.lfw_data import LFWDataset
import open_clip


def build_train_dataset(dataset_name, data_dir, annotation_path, augmentation):
    dataset = None

    if dataset_name == "vggface2":
        dataset = VGGFaceDataset(
            data_dir,
            annotation_path,
            augmentation=augmentation
        )
    elif dataset_name == "celeba":
        dataset = CelebADataset(
            data_dir,
            annotation_path,
            augmentation=augmentation
        )
    elif dataset_name == "ms1m":
        dataset = MS1M_Dataset(data_dir, augmentation)

    return dataset


def build_eval_dataset(dataset_name):
    _, transform, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-32-DataComp.M-s128M-b4K')
    return LFWDataset(transform=transform)

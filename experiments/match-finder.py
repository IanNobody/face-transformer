import sys
sys.path.append('../face-transformer')

import argparse
import torchvision.models as models
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from data.joined_dataset import JoinedDataset
import torchvision.transforms as T

def transforms():
    return T.Compose([
        T.ToTensor(),
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        # T.Normalize([0.48579846, 0.4084752675, 0.3738937391],
        #             [0.2683508996, 0.2441760085, 0.2375956649]),
        lambda x: x.to(torch.float32)
    ])

parser = argparse.ArgumentParser(
    description="A script for testing face recognition models on a dataset of images"
)

def load_model(model, checkpoint_path):
    if checkpoint_path:
        print("Loading checkpoint from ", checkpoint_path, "...")
        model.load_state_dict(torch.load(checkpoint_path)["model_weights"])

parser.add_argument("--ref", type=str, help="Path to reference dataset.", required=True)
parser.add_argument("--target", type=str, help="Path to target dataset.", required=True)
parser.add_argument("--checkpoint", type=str, help="Path to checkpoint.", required=True)
parser.add_argument("--threshold", type=float, help="Similarity threshold.", required=True)
parser.add_argument("--gpu", type=int, help="GPU device number.", required=True)
args = parser.parse_args()

model = models.swin_t()
model.head = nn.Linear(model.head.in_features, 512)
device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
model.to(device)

load_model(model, args.checkpoint)
dataset = JoinedDataset(args.target, args.ref, device=device, transform=transforms())
dataloader = DataLoader(dataset, batch_size=150, shuffle=False, collate_fn=JoinedDataset.collate_fn)
model.eval()
threshold = args.threshold

print("Dataset size: ", len(dataset))
print("Obtaining similarities...")

similarities = {}
matches = 0
for idx, data_sample in enumerate(dataloader):
    if idx % 1000 == 0:
        print(idx, "/", len(dataloader), file=sys.stderr)

    ref_imgs = data_sample["ref_image"].to(device)
    target_imgs = data_sample["target_image"].to(device)
    ref_cls = data_sample["ref_class"]
    target_cls = data_sample["target_class"]

    with torch.no_grad():
        ref_em = model(ref_imgs)
        tgt_em = model(target_imgs)

        for rf, tgt, rf_cm, tgt_c in zip(ref_em, tgt_em, ref_cls, target_cls):
            similarity = torch.cosine_similarity(rf, tgt, dim=0).item()
            if similarity > threshold:
                if tgt in similarities.keys():
                    print(tgt, " matches with ", rf_cm, " with similarity ", similarity)
                    record = (similarity, rf)
                    similarities[tgt].append(record)
                    matches += 1
                else:
                    similarities[tgt] = [(similarity, rf)]

print("Done, found ", matches, " matches.")
import os
from collections import OrderedDict
from datetime import time
import torch


def train(model, dataloader, optimizer, criterion, device, config):
    model.train()

    top_checkpoints = OrderedDict()
    for epoch in range(config.num_of_epoch):
        running_loss = 0.0

        for bidx, data in enumerate(dataloader):
            optimizer.zero_grad()

            x = data["image"]
            gt = data["entity"].detach()

            out = model(x)
            loss = criterion(out, gt)
            loss.backward()
            optimizer.step()

            current_loss = torch.mean(loss.item())
            running_loss += current_loss

            if bidx % config.sample_freq == config.sample_freq - 1:
                if len(top_checkpoints) < config.checkpoint_count or list(top_checkpoints.keys())[-1] < current_loss:
                    if len(top_checkpoints) >= config.checkpoint_count:
                        removed_checkpoint_suffix = top_checkpoints.popitem()
                        remove_checkpoint(config, removed_checkpoint_suffix)

                    new_checkpoint_suffix = get_checkpoint_suffix(epoch, bidx)
                    top_checkpoints[current_loss] = new_checkpoint_suffix
                    save_checkpoint(model, dataloader, config, new_checkpoint_suffix)


def get_checkpoint_suffix(epoch, bidx):
    time_str = time.strftime("%Y%m%d-%H%M%S")
    return time_str + "-epoch" + str(epoch) + "-batch" + str(bidx) + ".pth"


def remove_checkpoint(config, checkpoint_suffix):
    weight_path = config.weights_folder_path + "weight-" + checkpoint_suffix
    rng_path = config.weights_folder_path + "random-" + checkpoint_suffix
    os.remove(weight_path)
    os.remove(rng_path)


def save_checkpoint(model, dataloader, config, suffix):
    weight_path = config.export_weights_dir + "weight-" + suffix
    rng_path = config.export_weights_dir + "random-" + suffix
    print("****\nCheckpoint\n****\n")
    print("Saving wights to:", weight_path)
    torch.save(model.state_dict(), weight_path)
    sampler = dataloader.batch_sampler.sampler
    print("Saving random sampler state to:", rng_path, "\n****")
    torch.save(sampler.get_state(), rng_path)



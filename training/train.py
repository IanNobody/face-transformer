import os
from datetime import datetime
import torch
from os.path import join
import numpy


def train(model, dataloader, model_optimizer, loss_optimizer, criterion, device, config):
    model.train()

    top_checkpoints = []
    for epoch in range(config.num_of_epoch):
        running_loss = 0.0

        for bidx, data in enumerate(dataloader):
            model_optimizer.zero_grad()
            loss_optimizer.zero_grad()

            x = data["image"]
            gt = data["entity"].detach()

            if config.model_name == "dat":
                out = model(x)[0]
            else:
                out = model(x)

            loss = criterion(out, gt)
            loss.backward()
            model_optimizer.step()
            loss_optimizer.step()

            current_loss = loss.item()
            running_loss += current_loss

            if bidx % config.checkpoint_freq == config.checkpoint_freq - 1:
                checkpoint(model, dataloader, top_checkpoints, config, current_loss, epoch, bidx)

        checkpoint(model, dataloader, top_checkpoints, config, running_loss / len(dataloader), epoch, 0)
        print("Epoch ", epoch, " completed with loss: ", running_loss / len(dataloader))


def checkpoint(model, dataloader, top_checkpoints, config, loss, epoch, bidx):
    if len(top_checkpoints) >= config.checkpoint_count:
        removed_checkpoint_suffix = top_checkpoints.pop(0)
        remove_checkpoint(config, removed_checkpoint_suffix)

    new_checkpoint_suffix = get_checkpoint_suffix(epoch, bidx)
    top_checkpoints.append(new_checkpoint_suffix)
    save_checkpoint(model, dataloader, config, new_checkpoint_suffix)
    print("Epoch: ", epoch, " | Batch: ", bidx, " | Loss: ", loss, " | Checkpoint ID: ", new_checkpoint_suffix)


def get_checkpoint_suffix(epoch, bidx):
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    return time_str + "-epoch" + str(epoch) + "-batch" + str(bidx) + ".pth"


def remove_checkpoint(config, checkpoint_suffix):
    weight_path = config.export_weights_dir + "weight-" + checkpoint_suffix
    rng_path = config.export_weights_dir + "random-" + checkpoint_suffix
    os.remove(weight_path)
    os.remove(rng_path)


def save_checkpoint(model, dataloader, config, suffix):
    weight_path = join(config.export_weights_dir, "weight-" + suffix)
    rng_path = join(config.export_weights_dir, "random-" + suffix)
    torch.save(model.state_dict(), weight_path)
    sampler = dataloader.batch_sampler.sampler
    torch.save(sampler.get_state(), rng_path)



import os
from datetime import datetime
import torch
from os.path import join


def train(model, dataloader, model_optimizer, loss_optimizer, model_scheduler, loss_scheduler, criterion, device, config):
    model.train()

    top_checkpoints = []
    for epoch in range(config.num_of_epoch):
        running_loss = 0.0

        for bidx, data in enumerate(dataloader):
            model_optimizer.zero_grad()
            loss_optimizer.zero_grad()

            x = data["image"].to(device)
            gt = data["entity"].to(device)

            if config.model_name == "dat":
                out = model(x)[0]
            else:
                out = model(x)

            loss = criterion(out, gt)
            loss.backward()
            model_optimizer.step()
            loss_optimizer.step()
            model_scheduler.step()
            loss_scheduler.step()

            current_loss = loss.item()
            running_loss += current_loss

            if bidx % config.checkpoint_freq == config.checkpoint_freq - 1:
                checkpoint(model,
                           model_optimizer,
                           loss_optimizer,
                           dataloader,
                           criterion,
                           top_checkpoints,
                           config,
                           current_loss,
                           epoch,
                           bidx)

        checkpoint(model,
                   model_optimizer,
                   loss_optimizer,
                   dataloader,
                   criterion,
                   top_checkpoints,
                   config,
                   running_loss / len(dataloader),
                   epoch,
                   0,
                   permanent=True)
        print("Epoch ", epoch, " completed with loss: ", running_loss / len(dataloader))


def checkpoint(model,
               model_optimizer,
               criterion_optimizer,
               dataloader,
               criterion,
               top_checkpoints,
               config,
               loss,
               epoch,
               bidx,
               permanent=False):
    new_checkpoint = "checkpoint-" + str(epoch) + ".pth" if permanent else get_checkpoint_filename(epoch, bidx)

    if not permanent:
        if len(top_checkpoints) >= config.checkpoint_count:
            removed_checkpoint_suffix = top_checkpoints.pop(0)
            remove_checkpoint(config, removed_checkpoint_suffix)
        top_checkpoints.append(new_checkpoint)

    save_checkpoint(model, model_optimizer, criterion_optimizer, dataloader, criterion, config, new_checkpoint)
    print("Epoch: ", epoch, "| Batch: ", bidx, "| Loss: ", "{:.2f}".format(loss), "| Checkpoint file: ", new_checkpoint)


def get_checkpoint_filename(epoch, bidx):
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    return "checkpoint-" + time_str + "-epoch" + str(epoch) + "-batch" + str(bidx) + ".pth"


def remove_checkpoint(config, checkpoint_filename):
    os.remove(join(config.export_weights_dir, checkpoint_filename))


def save_checkpoint(model, model_optimizer, criterion_optimizer, dataloader, criterion, config, filename):
    torch.save({
        'model_weights': model.state_dict(),
        'model_optimizer': model_optimizer.state_dict(),
        'loss_optimizer': criterion_optimizer.state_dict(),
        'random_sampler': dataloader.batch_sampler.sampler.get_state(),
        'loss': criterion.state_dict(),
    }, join(config.export_weights_dir, filename))



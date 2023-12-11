import torch


def load_checkpoint(model, model_optimizer, loss_optimizer, criterion, random_sampler, config):
    if config.checkpoint_path:
        print("Loading checkpoint from ", config.checkpoint_path, "...")
        checkpoint = torch.load(config.checkpoint_path)
        model.load_state_dict(checkpoint["model_weights"])
        model_optimizer.load_state_dict(checkpoint["model_optimizer"])
        loss_optimizer.load_state_dict(checkpoint["loss_optimizer"])
        random_sampler.set_state(checkpoint["random_sampler"])
        criterion.load_state_dict(checkpoint["loss"])


def load_weights(model, checkpoint_path, device):
    if checkpoint_path:
        print("Loading checkpoint from ", checkpoint_path, "...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_weights"])
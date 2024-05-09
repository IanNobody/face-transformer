from models.wrappers import swin, resnet, flatten, smt, biformer, cmt, noisy_vit, openclip, multitask_openclip, old_multitask_openclip, old_swin
from models.wrappers.old_openclip import OldOpenCLIPWrapper


def build_model(config, embedding_size, num_of_classes):
    model = None
    if config.model_name == "swin":
        model = swin.SWINWrapper(embedding_size, num_of_classes)
    elif config.model_name == "resnet_50":
        model = resnet.ResNet50Wrapper(embedding_size, num_of_classes)
    elif config.model_name == "dat":
        raise NotImplementedError("DAT is no longer supported.")
        # model = dat.DATWrapper(embedding_size, num_of_classes)
    elif config.model_name == "flatten":
        model = flatten.FLattenWrapper(embedding_size, num_of_classes)
    elif config.model_name == "smt":
        model = smt.SMTWrapper(embedding_size, num_of_classes)
    elif config.model_name == "biformer":
        model = biformer.BiFormerWrapper(embedding_size, num_of_classes)
    elif config.model_name == "cmt":
        model = cmt.CMTWrapper(embedding_size, num_of_classes)
    elif config.model_name == "noisy_vit":
        model = noisy_vit.NoisyViTWrapper(embedding_size, num_of_classes)
    elif config.model_name == "openclip":
        model = openclip.OpenCLIPVisionWrapper(embedding_size, num_of_classes)
    elif config.model_name == "old_openclip":
        model = OldOpenCLIPWrapper(embedding_size, num_of_classes)
    elif config.model_name == "old_swin":
        model = old_swin.OldSWINWrapper(embedding_size, num_of_classes)
    elif config.model_name == "multitask_openclip":
        model = multitask_openclip.MultitaskOpenCLIPWrapper(embedding_size, num_of_classes)
    elif config.model_name == "old_multitask_openclip":
        model = old_multitask_openclip.OldMultitaskOpenCLIPWrapper(embedding_size, num_of_classes)

    if config.old_checkpoint_format:
        model.load_backbone_weights(config.weights_file_path)

    return model

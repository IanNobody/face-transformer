from models.wrappers import swin, resnet, flatten, smt, biformer, cmt, noisy_vit, openclip, multitask_openclip#, dat

def build_model(model_name, embedding_size, num_of_classes):
    model = None
    if model_name == "swin":
        model = swin.SWINWrapper(embedding_size, num_of_classes)
    elif model_name == "resnet_50":
        model = resnet.ResNet50Wrapper(embedding_size, num_of_classes)
    elif model_name == "dat":
        raise NotImplementedError("DAT is no longer supported.")
        # model = dat.DATWrapper(embedding_size, num_of_classes)
    elif model_name == "flatten":
        model = flatten.FLattenWrapper(embedding_size, num_of_classes)
    elif model_name == "smt":
        model = smt.SMTWrapper(embedding_size, num_of_classes)
    elif model_name == "biformer":
        model = biformer.BiFormerWrapper(embedding_size, num_of_classes)
    elif model_name == "cmt":
        model = cmt.CMTWrapper(embedding_size, num_of_classes)
    elif model_name == "noisy_vit":
        model = noisy_vit.NoisyViTWrapper(embedding_size, num_of_classes)
    elif model_name == "openclip":
        model = openclip.OpenCLIPVisionWrapper(embedding_size, num_of_classes)
    elif model_name == "multitask_openclip":
        model = multitask_openclip.MultitaskOpenCLIPWrapper(embedding_size, num_of_classes)

    return model

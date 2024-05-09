from models.model_factory import build_model
from train_utils.train_config import TrainingConfiguration
import torch
import time

IMG_SIZE = 224
NUM_TESTS = 100

def test_speed(model, num_tests=1000, input_size=(224, 224)):
    dummy_input = torch.randn(1, 3, *input_size)
    if torch.cuda.is_available():
        dummy_input = dummy_input.to(device)

    for _ in range(100):
        _ = model(dummy_input)

    start_time = time.time()
    for _ in range(num_tests):
        with torch.no_grad():
            _ = model(dummy_input)
    end_time = time.time()
    return start_time, end_time


swin_cfg = TrainingConfiguration(model_name="swin", dataset_name="")
flat_cfg = TrainingConfiguration(model_name="flatten", dataset_name="")
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

swin = build_model(swin_cfg, 512, 100000)
swin.eval()
if torch.cuda.is_available():
    flat = swin.to(device)

print("Testing SWIN")
print("_________________")
swin_start_time, swin_end_time = test_speed(swin, NUM_TESTS, (IMG_SIZE, IMG_SIZE))
del swin
torch.cuda.empty_cache()

flat = build_model(flat_cfg, 512, 100000)
flat.eval()
if torch.cuda.is_available():
    flat = flat.to(device)

print("Testing FLatten")
print("_________________")
flat_start_time, flat_end_time = test_speed(flat, NUM_TESTS, (IMG_SIZE, IMG_SIZE))

swin_time = ((swin_end_time - swin_start_time) / NUM_TESTS) * 1000
flat_time = ((flat_end_time - flat_start_time) / NUM_TESTS) * 1000

print("****************************")
print("Image resolution: ", IMG_SIZE)
print("****************************")
print(f"AVG infer. time - SWIN: {swin_time:.6f} ms")
print(f"AVG infer. time - FLatten: {flat_time:.6f} ms")
print("****************************")
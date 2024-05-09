
# Vision Transformers for Facial Recognition
## Brno University of Technology - Faculty of Informatics

****Author:**** Šimon Strýček <[xstryc06@stud.fit.vutbr.cz](mailto:xstryc06@stud.fit.vutbr.cz)>
****Supervisor:**** Ing. Jakub Špahňel

---
This document contains general information about the master thesis and manual how to use the code provided.

---

### Requirements

To use the provided scripts, you need to have Python version 3.10+ installed.
The required packages can be found in `requirements.txt` file. 
Some parts of the code are taken from other source (neural network models and some of the evaluation implementation).
Such implementation is marked directly in the file header.

### Usage

To execute either training implementation or validation, you can use `main.py` script present in the root `src/` directory.

The `main.py` interface has the following options:

`--model <NAME>` -- selects the NN implementation to use (swin, resnet_50, flatten, set, biformer, cmt, noisy_vit, openclip, multitask_openclip)
`--dataset <NAME>` -- selects training dataset to use - needs to be used even when running validation (egg, celeba, ms1m)
`--data_path <PATH>` -- path to images
`--annotation_dir <PATH>` -- path to dir containing dataset annotations (not needed when using datasets without direct annotations - MS1Mv3)
`--checkpoints_dir <PATH>` -- path where to save checkpoints when training
`--weights_file_path <PATH>` -- path to the checkpoint file to load; in case of validation, a directory containing multiple files is expected
`-b <NUM>` -- minibatch size
`-e <NUM>` -- number of training epochs
`--gpu <[NUM]>` -- index(es) of GPU(s) to use; can be multiple
`--output_dir <PATH>` -- path where to save generated plots when running validations

**Training example:**
```
python main.py --model swin --dataset ms1m --data_path /data/images/ --checkpoints_dir /checkpoints/swin/ -b 200 -e 100 --gpu 0 1 2
```  

**Validation example:**
```
python main.py --model swin --dataset ms1m --data_path /data/images --weights_file_path /checkpoints/swin/ -b 200 --gpu 0 --output_dir /val/plots/
```


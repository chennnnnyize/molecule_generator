# Conditional Molecule Generator
This repository contains the source code and data sets for the graph based molecule generator discussed in the article "Multi-Objective De Novo Drug Design with Conditional Graph Generative Model" (https://arxiv.org/abs/1801.07299).

## Requirement

The model is built using Python 2.7, and utilizes the following packages:

- Tensorflow 1.3
- RDKit
- Numpy
- networkx 2.0

## Todo list

The current repo contains only data sets and the source code, future updates will include:

- [ ] Models trained using the full dataset
- [ ] Add file descriptions and tutorials
- [ ] Activity data and trained predictor for GSK3b and JNK3

## Usage

Run the `train.py` file for the network training. Set the environment variable `TF_CPU_ALLOCATOR_USE_BFC`

to `true` to avoid memory leak in CPU. 

To train on a single GPU, use the `single` command:

```shell
python train.py single [config_dir]
```

Where `[config_dir]` is the directory containing model config files. Default configs are provided in the `models` folder (description of those files will be contained in the future release). For training with multiple GPUs, use `multiple` command:

```shell
# start parameter server
CUDA_VISIBLE_DEVICES="" nohup python train.py multiple [config_dir] ps 0

# start generator
CUDA_VISIBLE_DEVICES="" nohup python train.py multiple [config_dir] generator 0
CUDA_VISIBLE_DEVICES="" nohup python train.py multiple [config_dir] generator 1

# start worker
CUDA_VISIBLE_DEVICES=0 nohup python train.py multiple [config_dir] worker 0
CUDA_VISIBLE_DEVICES=1 nohup python train.py multiple [config_dir] worker 2
...
CUDA_VISIBLE_DEVICES=[n-1] nohup python train.py multiple [config_dir] worker [n-1]
```

Where `n` is the number of GPUs used during training. The default value for `n` is 4. To change `n`,  you need to modify the corresponding config file.

To train with the entire dataset, use `single-full` and `multiple-full` respectively for single and multiple GPU.

It should be reminded that the default config requires a large amount of memory for training. Limit the batch size or `k` in the config file if the memory resource is limited.

Contact me if you have any questions.
Email: 1210307427@pku.edu.cn or kevinid4g@gmail.com




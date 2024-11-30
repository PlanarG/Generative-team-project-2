# UNet DDPM on quickdraw dataset

This repository contains the code for training a UNet model with DDPM on the quickdraw dataset. 

## Pre-requisites

The quickest way to install the dependencies is to run the following command:

```
pip install -r requirements.txt
```

## Training

The quickdraw dataset is available at https://github.com/googlecreativelab/quickdraw-dataset. The dataset should be downloaded and extracted to the `dataset` directory. The dataset should be structured as follows:

```
dataset
├── airplane
│   ├── 0.png
│   ├── 1.png
│   └── ...
├── alarm clock
│   ├── 0.png
│   ├── 1.png
│   └── ...
└── ...
```

To train the model, run the following command in the root directory of the repository:

```bash
python src/train.py
```

This will train the model defined in `src/model.py` on the quickdraw dataset with the configuration defined in `config/config.yml`. Predicted images will be saved in the `output` directory. 
The config file should be structured like 

```yaml
image_size: 64 # image size
batch_size: 64 # training batch size
eval_seed: 175 # seed for evaluation
eval_batch_size: 8 # evaluation batch size
num_epochs: 50 # number of epochs
num_training_steps: 1000 # number of training timesteps for noise schedule
num_inference_steps: 100 # number of inference timesteps for noise schedule
save_every_epoch: 1 # saving interval
logger: wandb # logger, this line is optional

optimizer: AdamW # optimizer defined in torch.nn.optim
optimizer_params: # optimizer parameters
  weight_decay: 0 

dataset_dir: dataset # directory for dataset
checkpoint_dir: checkpoint # directory for saving checkpoints
load_checkpoint: model_2024-11-29_16-02-19.pt # checkpoint to load, this line is optional
output_dir: output # directory for saving output
scheduler: OneCycleLR # scheduler defined in torch.optim.lr_scheduler
lr_initial: 1.e-5 
max_lr: 5.e-5
total_steps: 30000
pct_start: 0.001
anneal_strategy: linear
clip_grad_norm: 1.0    
```

## Evaluation

To evaluate the model, run the following command in the root directory of the repository:

```bash
python src/eval.py
```

The ouput images will be saved to the `output` directory.
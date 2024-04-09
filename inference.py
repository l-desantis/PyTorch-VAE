import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
from pytorch_lightning.plugins import DDPPlugin

def print_maze_cute(maze):
    string = ""
    conv = {
        3: "E ",
        2: "S ",
        1: "██",
        0: "  "
    }
    for y in range(len(maze)):
        for x in range(len(maze[y])):
            string += conv[maze[y][x]]
        string += "\n"
    print(string)

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

model = vae_models[config['model_params']['name']](**config['model_params'])
ckpt = torch.load('logs/VanillaVAE/version_0/checkpoints/last.ckpt')
experiment = VAEXperiment(model, config['exp_params'])
experiment.load_state_dict(ckpt['state_dict'])

print(f"======= Inference {config['model_params']['name']} =======")
res = experiment.sample(n_samples=10)
res = torch.squeeze(res, 1).round().detach().cpu().numpy()
print_maze_cute(res[0])


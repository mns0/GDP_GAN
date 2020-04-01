import numpy as np
import torch.onnx
from torch import nn
from models_and_dataloader import Generator_RNN
from models_and_dataloader import LockedDropout



#load the model
model_name = './models2/alt_netG_e1900.pth'
batch_size = 1

kwagsG = {"target_size": 1, "predictor_dim": 10, "hidden_dim": 200,
          "num_layers": 1, "dropout_prob": 0.4, 'train': False}

check_gen = Generator_RNN(**kwagsG)
check_gen = torch.load(dir_model+gen_model, map_location=device)

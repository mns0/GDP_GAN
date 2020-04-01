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


#initalized mode with pretrained weights
device = torch.device('cpu')
gen = Generator_RNN(**kwagsG)
gen = torch.load(model_name, map_location=device)
batch_size, seq_len = 20, 15

gen.eval()
noise_x = torch.randn(batch_size, seq_len, 1, requires_grad=True)
noise_y = torch.randn(batch_size, seq_len, 9, requires_grad=True)


model_out = gen(noise_x,noise_y)
torch.onnx.export(gen, (noise_x,noise_y), "gen_8020_model.onnx", verbose=True) 


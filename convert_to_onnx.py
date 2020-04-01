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
noise_x = torch.randn(batch_size, seq_len, 1)
noise_y = torch.randn(batch_size, seq_len, 9)


model_out = gen(noise_x,noise_y)
torch.onnx.export(gen,               # model being run
                  (noise_x,noise_y),                         # model input (or a tuple for multiple inputs)
                  "gen_8020_model.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})


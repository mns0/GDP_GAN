# get the data
import matplotlib.pyplot as plt
import numpy as np
import torch
#import __main__
from models_and_dataloader import GDPData
from models_and_dataloader import Generator_RNN
from models_and_dataloader import LockedDropout
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-talk')

dir_data, dir_model = './', './models2/'
train_data = 'Preprocessed_data_gdp_pc.dat'
start, end, stride = 0, 2000, 100
epoches, batch_size, seq_len = 200, 20, 15

kwagsG = {"target_size": 1, "predictor_dim": 10, "hidden_dim": 200,
          "num_layers": 1, "dropout_prob": 0.4, 'train': False}

device = torch.device('cpu')
gdp_dataset = GDPData(dir_data + train_data, normalize=True)
x, y  = gdp_dataset.x.squeeze(-1), gdp_dataset.y.unsqueeze(2)
_min, _max = gdp_dataset.normalize_table[-1]

#get generator data
generated_time_series = []
for i in range(start,end,stride):
    gen_model = f'alt_netG_e{i}.pth'
    check_gen = Generator_RNN(**kwagsG)
    check_gen = torch.load(dir_model+gen_model, map_location=device)
    check_gen.eval()
    gen_ts = []
    end_idx = (len(x)//batch_size)*batch_size
    for cond in np.array_split(x[:end_idx],len(x)//batch_size):
      if cond.shape[0] != batch_size:
        continue
      noise_x = torch.randn(batch_size, seq_len, 1)
      gen_data = check_gen(noise_x,cond).detach().numpy().reshape(-1)
      gen_ts.extend(gen_data.reshape(-1))
    
    #unnormalize data
    gen_ts = np.asarray(gen_ts)
    gen_ts = 0.5* (gen_ts*_max - gen_ts*_min + _max + _min)
    generated_time_series.append(gen_ts)


#created an animated graph
ts_len = len(generated_time_series[0])
fig = plt.figure()
ax = plt.axes(xlim=(0,ts_len) , ylim=(-3, 6))
line, = ax.plot([], [], lw=3)

real_y = y.squeeze(2).numpy().reshape(-1)
real_y = 0.5* (real_y*_max - real_y*_min + _max + _min)
ax.plot(np.arange(len(real_y)),real_y)

def init():
    line.set_data([], [])
    return line,
def animate(i):
    x = np.arange(ts_len)
    y = generated_time_series[i]
    line.set_data(x, y)
    return line,


ax.set_ylabel("Quartly percent difference")
ax.set_ylabel("Date")

fnum = int((end-start)//stride)
anim = FuncAnimation(fig, animate, init_func=init,
                               frames=fnum, interval=200, blit=True)

anim.save('partial80-20_model.gif', writer='imagemagick')

#
#fig, ax = plt.subplots(1,1)
#ax.plot(np.arange(len(gen_ts)),gen_ts)
#ax.plot(np.arange(len(real_y)),real_y)
#
#ax.set_ylim((-2,6))
#ax.set_ylim((-2,6))

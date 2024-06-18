import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim  # 隐层大小
        self.num_layers = num_layers  # LSTM层数
        # input_dim为特征维度，就是每个时间点对应的特征数量，这里为14
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, h_n = self.gru(x)  # output为所有时间片的输出，形状为：16,1,4
        # print(output.shape) torch.Size([16, 1, 64]) batch_size,timestep,hidden_dim
        # print(h_n.shape) torch.Size([3, 16, 64]) num_layers,batch_size,hidden_dim
        # print(c_n.shape) torch.Size([3, 16, 64]) num_layers,batch_size,hidden_dim
        batch_size, timestep, hidden_dim = output.shape

        # 将output变成 batch_size * timestep, hidden_dim
        output = output.reshape(-1, hidden_dim)
        output = self.fc(output)  # 形状为batch_size * timestep, 1
        output = output.reshape(timestep, batch_size, -1)
        return output[-1]  # 返回最后一个时间片的输出




input_str='62,126,13,0.8,59,125#18,31,8,0.7,47,161'
Predict_data = str(input_str.replace("'", '')).split('#')
fitness_list = []
for i in range(len(Predict_data)):
     fitness_list .append(Predict_data[i].split(','))
a = np.array(fitness_list)
txx = pd.DataFrame(a[:, 0:np.shape(a)[1]])
min_maxs = np.recfromtxt('./data/mp2.5_param.txt')
base_mean = np.split(min_maxs, 2, axis=0)[0]
base_std = np.split(min_maxs, 2, axis=0)[1]

# txx1 = (txx.astype('float') - base_mean) / base_std
txx1 = (txx.astype('float') - base_mean[:len(base_mean) - 1]) / base_std[:len(base_std) - 1]
x0 = torch.from_numpy(txx1.values).unsqueeze(1).float()

model_name = 'gru'
save_path = './{}.pt'.format(model_name)




input_dim = 6  # 每个步长对应的特征数量，就是使用每天的4个特征，最高、最低、开盘、落盘
hidden_dim = 120  # 隐层大小
num_layers = 2  # LSTM的层数
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GRU(input_dim, hidden_dim, num_layers, 1).to(device)
model.load_state_dict(torch.load(save_path))
y_test_pred = model(x0)
print(y_test_pred)
print(base_std[len(base_std) - 1:])
print(base_mean[len(base_mean) - 1:])


predictions = {}
prediction_np = y_test_pred.data.numpy().flatten()
y=prediction_np*base_std[len(base_std) - 1:]+base_mean[len(base_mean) - 1:]
a = y.tolist()
str_result=''
for temp in range(len(a)):
    str_result = str_result + str(a[temp])  + '#'
str_result = str_result.strip('#')
print(str_result)
# tensor([[0.3073]], grad_fn=<SelectBackward0>)
# AQI    31.1778
# dtype: float64
# AQI    78.522222

# for model in models:
#     y_test_pred = model(x0)
#     predictions_v.append(y_test_pred)
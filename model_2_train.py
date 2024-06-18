import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import visdom


# viz = visdom.Visdom(port=8097, server="127.0.0.1",env="gruAQI")
# # line updates
# loss_win = viz.line(np.arange(1))

df = pd.read_csv("./data/pm2.5.csv",encoding="utf-8")
df.head()

#删除没有用的字段
del df["质量等级"]
#Z-Score标准换
features = df[["PM2.5","PM10","SO2","CO","NO2","O3_8h"]]
df[["PM2.5","PM10","SO2","CO","NO2","O3_8h"]] = (features-features.mean())/features.std()

features = df[["PM2.5","PM10","SO2","CO","NO2","O3_8h"]]
df[["PM2.5","PM10","SO2","CO","NO2","O3_8h"]] = (features-features.mean())/features.std()
df.head()
df.count()
df.describe()
train_df= df[:300]
test_df = df[300:]



index = range(0,df.count()[0])
plt.figure(figsize=(12,6))
plt.ylabel("AQI")
plt.xlabel("TIME")
plt.plot(range(0,300),df["AQI"][:300],"-.")
plt.plot(range(300,360),df["AQI"][300:])
plt.legend(["Train","Test"])
plt.show()


bins = 60            # RNN时间步长
input_dim = 6       # RNN输入尺寸
lr = 0.001            # 初始学习率
epochs = 1000        # 轮数
hidden_size=128       # 隐藏层神经元个数
num_layers = 2       # 神经元层数
bidirectional = True #双向循环
batch_size = 1

class GRUAQI(nn.Module):
    def __init__(self,input_dim,hidden_size,num_layers,bidirectional,power):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional
        )
        self.out = nn.Linear(hidden_size*power,1)

    def forward(self, x, h):
        r_out, h_state = self.gru(x,h)
        outs = []
        for record in range(r_out.size(1)):
            out = self.out(r_out[:, record, :])
            outs.append(out)
        return torch.stack(outs, dim=1), h_state


device = "cuda" if torch.cuda.is_available() else "cpu"
# if bidirectional=True
power = 2 if (bidirectional) else 1

gruAQI = GRUAQI(input_dim, hidden_size, num_layers, bidirectional, power).to(device)
optimizer = torch.optim.Adam(gruAQI.parameters(), lr=lr)
loss_func = nn.MSELoss()

c_state = torch.zeros(num_layers * power, batch_size, hidden_size).to(device)
global_step = 0
for step in range(epochs):
    windows = int(np.ceil(300 / bins))
    for window in range(windows):
        steps = train_df[window * bins:(window + 1) * bins]
        x = torch.from_numpy(steps[["PM2.5", "PM10", "SO2", "CO", "NO2", "O3_8h"]].values).unsqueeze(1).float().to(
            device)
        # 【bins，batch,input_size=6】(50,1,6)
        y = torch.from_numpy(steps["AQI"].values).unsqueeze(1).unsqueeze(2).float().to(
            device)
        # 【bins，batch,output=1】(50,1,1)
        prediction, c_state = gruAQI(x, c_state)  # RNN输出（预测结果，隐藏状态）
        # 将每一次输出的中间状态传递下去(不带梯度)
        c_state = c_state.detach()
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        if (global_step % 100 == 0):
            # viz.line(Y=np.array([loss.item()]), X=np.array([global_step]), update='append', win=loss_win)
            print("loss:{:.8f}".format(loss))

def predict(model,input_data):
    length = input_data.count()[0]
    outs = []
    windows = int(np.ceil(length/bins))
    for window in range(windows):
        prediction, c_state = gruAQI(torch.from_numpy(input_data[["PM2.5","PM10","SO2","CO","NO2","O3_8h"]]
                              [window*bins:(window+1)*bins].values).unsqueeze(1).float().to(device),None)
        outs.extend(prediction.cpu().data.numpy().flatten())
    return outs


plt.figure(figsize=(15,5))
plt.plot(list(range(0,360)),df["AQI"].values,"-.")
plt.plot(list(range(300,360)),predict(gruAQI,test_df))
print("Test Loss:{:.8f}".format(loss_func(torch.FloatTensor(predict(gruAQI,test_df)), torch.from_numpy(test_df["AQI"].values).float())))
plt.legend(["Raw","Predict"])
plt.show()
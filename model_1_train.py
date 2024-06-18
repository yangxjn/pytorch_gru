import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset


# Numpy is not available  错误时，请更新numpy==1.26.1

# 按80%训练，10进行监督

# 1.定义GRU网络
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

# seq_length = 1  # 时间步长，就是利用多少时间窗口
def split_data(data, seq_length,input_dim):
    # predict_num 预测几个结果的各数，比如 产油和产水
    predict_num=1
    coloum_num=data.shape[1]-predict_num

    dataX = []  # 保存X
    dataY = []  # 保存Y
    # 将整个窗口的数据保存到X中，将未来一天保存到Y中
    for index in range(len(data) - seq_length):
        dataX.append(data[index: index + seq_length,:coloum_num])
        dataY.append(data[index: index + seq_length,coloum_num:])

    dataX = np.array(dataX)
    dataY = np.array(dataY)

    # 获取训练集大小
    train_size = int(np.round(0.8 * dataX.shape[0]))

    # 划分训练集、测试集
    # x:【bins，batch,input_size=6】(50,1,6)
    # y:【bins，batch,output=1】(50,1,1)
    x_train = dataX[: train_size, :].reshape(-1, seq_length, input_dim)
    y_train = dataY[: train_size].reshape(-1, seq_length, predict_num)

    x_test = dataX[train_size:, :].reshape(-1, seq_length, input_dim)
    y_test = dataY[train_size:].reshape(-1, seq_length, predict_num)

    return [x_train, y_train, x_test, y_test]

input_str='62,126,13,0.8,59,125'
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

seq_length = 1  # 时间步长，就是利用多少时间窗口
batch_size = 60  # 批次大小,训练模型是往模型一次送入的记录数个数。
input_dim = 6  # 每个步长对应的特征数量，就是使用每天的4个特征，最高、最低、开盘、落盘
predict_num=1  #预测内容的数据各数
hidden_dim = 120  # 隐层大小
output_dim = 1  # 由于是回归任务，最终输出层大小为1
num_layers = 2  # LSTM的层数
epochs = 1000
best_loss = float('inf')
model_name = 'gru'
save_path = './{}.pt'.format(model_name)


df= pd.read_excel("./data/pm2.5.xls", index_col=0)
# reverse_data = ori_data[::-1]
# pd=reverse_data.iloc[:, 1:].values
df.head()


pd=df.iloc[:]
#Z-Score标准换

str_filename="./data/mp2.5_param.txt"

base_mean=pd.mean();
base_std=pd.std();
AQI_mean=base_mean[len(base_mean) - 1:]
AQI_std=base_std[len(base_std) - 1:]

np.savetxt(str_filename, (np.append(base_mean, base_std)), delimiter=',', fmt='%0.2f')

df1 = (pd-base_mean)/base_std
df2 = df1.values


# 3.获取训练数据   x_train: 300,1,6
x_train, y_train, x_test, y_test = split_data(df2, seq_length, input_dim)
# 4.将数据转为tensor
print(type(x_train))
print(type(y_train))
x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
y_test_tensor = torch.from_numpy(y_test).to(torch.float32)

# 5.形成训练数据集
train_data = TensorDataset(x_train_tensor, y_train_tensor)
test_data = TensorDataset(x_test_tensor, y_test_tensor)

# 6.将数据加载成迭代器
train_loader = torch.utils.data.DataLoader(train_data,  batch_size, True)

test_loader = torch.utils.data.DataLoader(test_data, batch_size, False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GRU(input_dim, hidden_dim, num_layers, 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.MSELoss()  # 定义损失函数
c_state = torch.zeros(num_layers,batch_size,hidden_dim).to(device)
# 8.模型训练
recode_loss = []  # 保存X
for epoch in range(epochs):
    model.train()
    running_loss = 0
    train_bar = tqdm(train_loader)  # 形成进度条
    for data in train_bar:
        x_train, y_train = data  # 解包迭代器中的X和Y
        optimizer.zero_grad()
        y_train_pred = model(x_train)
        loss = loss_function(y_train_pred, y_train.reshape(-1, 1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 epochs,
                                                                 loss)

    # 模型验证
    model.eval()
    test_loss = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for data in test_bar:
            x_test, y_test = data
            y_test_pred = model(x_test)
            test_loss = loss_function(y_test_pred, y_test.reshape(-1, 1))

    if test_loss < best_loss:


        best_loss = test_loss
        torch.save(model.state_dict(), save_path)
        recode_loss.append(test_loss.numpy())


print('Finished Training')
print(best_loss)


# 9.绘制结果
plt.figure(figsize=(12, 8))
plt.plot(recode_loss, "b")
plt.legend()
plt.show()
# 9.绘制结果
# plt.figure(figsize=(12, 8))
# plt.plot(scaler.inverse_transform((model(x_train_tensor).detach().numpy()).reshape(-1, 1)), "b")
# plt.plot(scaler.inverse_transform(y_train_tensor.detach().numpy().reshape(-1, 1)), "r")
# plt.legend()
# plt.show()
#
# y_test_pred = model(x_test_tensor)
# plt.figure(figsize=(12, 8))
# plt.plot(scaler.inverse_transform(y_test_pred.detach().numpy()), "b")
# plt.plot(scaler.inverse_transform(y_test_tensor.detach().numpy().reshape(-1, 1)), "r")
# plt.legend()
# plt.show()
y_test_pred = model(x0)
model.eval()
print(y_test_pred)
print(AQI_std)
print(AQI_mean)
prediction_np = y_test_pred.data.numpy().flatten()
y=prediction_np*AQI_std+AQI_mean
print(y)
model_name = 'gru_good'
save_path = './{}.pt'.format(model_name)
torch.save(model.state_dict(), save_path)
print('Finished Training 0ver')
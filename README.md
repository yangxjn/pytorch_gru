# pytorch_gru
predict 预测模型中，当单条记录进行预测时，需要对batch_size=1  seq_length=1 设置,其中假设input_dim=6元素输入时，输入数据体结构为：[batch_size,seq_length,input_dim]=>[1,1,6]
train 训练模型中，由于模型复杂度较低，数据量较小，所有epoch尽量的小，batch_size设置尽量的大。

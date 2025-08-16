import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as DATA
from raw_single_yueshu import BPGNN
import os
from sklearn.preprocessing import MinMaxScaler
import time

#超参数设置:见iPad
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
gpu = '0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
if torch.cuda.is_available():
    print('gpu可用')
device = torch.device('cuda:0')#if torch.cuda.is_available() else 'cpu'

# 数据集的加载
scaler1 = MinMaxScaler(feature_range=(0,1))#将数据的每一个特征缩放到给定的范围，将数据的每一个属性值减去其最小值，然后除以其极差（最大值 - 最小值）
data_frame = pd.read_csv('shou_zhanxiang_bishe_yueshu_input.csv', header=None, engine='python',encoding = 'gb2312')#header=None是无表头的意思，header=0添加表头即第一行，列没有表头
dataset = data_frame.values#不可以删除，是将点云格式的数据转换为数组格式， values()函数是Python字典(Dictionary)的一个方法，它返回一个字典中所有的值
# print(dataset)
dataset = scaler1.fit_transform(dataset)#归一化，只要归到0-1之间就行，线性归一化也行，且归一化是按列归一化
# print(dataset)
label_frame = pd.read_csv('shou_zhanxiang_bishe_yueshu_output.csv', header=None, engine='python',encoding = 'gb2312')
labelset = label_frame.values
# print(labelset)
labelset = scaler1.fit_transform(labelset)
# print(labelset)
x_train, x_test, y_train, y_test = train_test_split(dataset, labelset, test_size=0.2, shuffle=True, random_state=19)#划分数据集，并打乱（但每次拆分的结果是相同的）
# print(y_train)
train_xt = torch.from_numpy(x_train.astype(np.float32)).to(device)#将训练集输入数据转换为 PyTorch Tensor张量形式，并将其移动到指定的设备（device）上。
train_yt = torch.from_numpy(y_train.astype(np.float32)).to(device)
test_xt = torch.from_numpy(x_test.astype(np.float32)).to(device)
test_yt = torch.from_numpy(y_test.astype(np.float32)).to(device)

train_data = DATA.TensorDataset(train_xt, train_yt)#data.TensorDataset的主要作用是将输入的张量组合成一个数据集，使得在训练过程中可以方便地进行数据加载和迭代。
test_data = DATA.TensorDataset(test_xt, test_yt)
train_loader = DATA.DataLoader(dataset=train_data,batch_size=50)#batch_size一批样本的大小， train_loader的作用就是把所有数据分成几个批次，对每个批次进行训练

def nettrain (para_epoch): #执行完一次前向传播和反向传播叫做一个epoch
    # 神经网络训练，可重复使用
    subGNN = BPGNN().to(device)
    optimizer = torch.optim.Adam(subGNN.parameters(),lr=0.01)#创建一个 Adam 优化器
    lrr = torch.optim.lr_scheduler.StepLR(optimizer,step_size=200, gamma=0.1)#创建一个学习率调度器（StepLR 调度器），首先一个大前提是需要自己定义一个model和一个optimizer, 最后再将 optimizer 放入 lr_scheduler 中（前两行），
    loss_func = nn.MSELoss()#创建均方误差（MSE）损失函数，它是实际观察值与预测值之差的平方和的平均值。
    train_loss_all = []#创建一个列表，用于存储每个epoch的训练损失。
    for epoch in range(para_epoch):
        # if epoch%100==0:
        #     print(epoch)
        train_loss = 0
        train_num = 0
        for step, (b_x, b_y) in enumerate(train_loader): #从train_loader提取步数，特征值和标签，共执行N/train_loader次，N为总样本数，enumerate(train_loader)会遍历train_loader中的每个批次。
            #print(b_x)
            output = subGNN(b_x)
            # regular_loss=0
            # for param in subGNN.parameters():
            #     regular_loss+=(param**2).sum()
            loss = loss_func(output, b_y)#+0.0001*regular_loss
            optimizer.zero_grad()#梯度清零
            loss.backward()#反向传播
            optimizer.step()#权重等参数的调整
            train_loss += loss.item() * b_x.size(0)#b_x.size(0)为一批的算例数，b_x.size(1)为特征数量
            train_num += b_x.size(0)
        train_loss_all.append(train_loss / train_num)#train_loss / train_num为平均损失，将这个平均损失添加到 train_loss_all 列表中，用于后续分析或可视化训练过程中的损失。
        lrr.step()#更新优化器中的学习率
    # 误差测试
    pre_y = subGNN(train_xt)#train_xt为 PyTorch Tensor张量形式，
    pre_y = pre_y.data.cpu().numpy()#将预测结果 pre_y 从GPU移动到CPU，并将其转换为NumPy数组，对数组进行运算
    pre_y_inverse = scaler1.inverse_transform(pre_y)
    y_train_inverse = scaler1.inverse_transform(y_train)
    # print(pre_y_inverse)
    # print(y_train_inverse)
    mae_train = np.average(np.abs(pre_y_inverse-y_train_inverse)/y_train_inverse) #skm.mean_absolute_error(y_train, pre_y)   #np.abs(pre_y-y_train)/y_train将数组取平均值
    print('训练集上的相对误差为', mae_train)
    pre_y = subGNN(test_xt)
    pre_y = pre_y.data.cpu().numpy()
    pre_y_inverse = scaler1.inverse_transform(pre_y)
    y_test_inverse = scaler1.inverse_transform(y_test)
    # print(pre_y_inverse)
    # print(y_text_inverse)
    mae_test = np.average(np.abs(pre_y_inverse-y_test_inverse)/y_test_inverse)#np.average(np.abs(pre_y-y_test)/y_test)
    print('测试集上的相对误差为', mae_test)
    print('………………')
    return subGNN, train_loss_all, mae_train, mae_test

if __name__ == '__main__':
    xxx=[]
    train_mae = []
    test_mae = []
    minloss=100000
    start = time.perf_counter()
    for i in range(1): #开始一个循环，迭代5次(0-4)
        subGNN, train_loss_all, mae_train, mae_test= nettrain(2000)
        xxx.append(i) #在列表中添加一个元素（相加作用）
        train_mae.append(mae_train)
        test_mae.append(mae_test)
        if mae_train+mae_test<minloss:
            minloss=mae_train+mae_test
            bestnet=subGNN
    print("运行5次平均误差为：",np.mean(test_mae))#np.mean()是NumPy库的函数，用于计算数组中所有元素的平均值
    torch.save(bestnet,"1123323.pkl")
    end = time.perf_counter()
    runTime = end - start
    print("运行时间：", runTime)
    # torch.save(bestnet, "OLDs5.pkl")

    # oldnet = torch.load(".\\OLDs5.pkl").cpu()
    # for x in
    #     input1 = torch.Tensor(np.array([1, 2, 3, 4, 5]))
    # output1 = newnet(input1)
    '''
    #误差可视化2
    plt.figure()
    plt.plot(xxx,train_mae, "ro-", label='Train Loss')
    plt.plot(xxx,test_mae, "bo-", label="Test Loss")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()
    '''
    '''
    pltxx = torch.from_numpy(dataset.astype(np.float32))
    yyy=bestnet(pltxx).data.numpy()
    yyy=scaler2.inverse_transform(yyy)
    labelset=scaler2.inverse_transform(labelset)
    plt.figure()
    plt.plot(yyy,"ro-", label='pre')
    print(yyy)
    plt.plot(labelset,"bo-", label='real')
    plt.legend()
    plt.grid()
    plt.ylabel("flow")
    plt.xlabel("number")
    plt.show()
    '''

    # # 训练过程可视化
    # subGNN, train_loss_all, mae_train, mae_test = nettrain(500)
    # plt.figure()
    # plt.plot(train_loss_all, "ro-", label='Train Loss')
    # plt.legend()
    # plt.grid()
    # plt.ylabel("Loss")
    # plt.xlabel("epoch")
    # plt.show()
    # torch.save(subGNN, "OLDNET01.pkl")

    '''
    netload=torch.load(".\\bestnet1pro.pkl")
    makedata_frame = pd.read_csv('makedata.csv', header=None, engine='python')
    makedataset = makedata_frame.values
    makedataset = scaler1.fit_transform(makedataset)
    pltxx = torch.from_numpy(makedataset.astype(np.float32))
    yyy=netload(pltxx).data.numpy()
    yyy=scaler2.inverse_transform(yyy)
    f = open('datamaked.csv','w',newline="")
    csv_writer = csv.writer(f)
    for i in range(len(yyy)):
        csv_writer.writerow([yyy[i][0]])
    f.close()
    '''
    '''
    netload=torch.load(".\\bestnet1.pkl")
    pre_y = netload(train_xt)
    pre_y = pre_y.data.numpy()
    mae_train = np.average(np.abs(pre_y-y_train)/y_train)
    print(mae_train)
    pre_y = netload(test_xt)
    pre_y = pre_y.data.numpy()
    mae_test = np.average(np.abs(pre_y-y_test)/y_test)
    print(mae_test)
    '''
import numpy as np
import torch
from scipy import interpolate
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import math
import csv
import matplotlib.cm as cm
import pandas as pd
import tecplot
from tecplot.constant import *

#读入各处展向分布的参数
a1 = [1, 0.5, 0.5, 0, 0.5]

#返回预测值_8000（最大最小值不同）
def predict (n,m):
    newnet = torch.load(".\\NEW_收敛孔_二维_直接预测_从xd1开始_8000.pkl").cpu()
    x_d = (n-1) / 29
    y_d = (m+3.5) / 7
    input1 = torch.Tensor(np.array([a1[0], a1[1], a1[2], a1[3], a1[4], x_d, y_d])) #归一化后的数据
    out1 = newnet(input1)
    output1 = out1.item()
    # print(output1)
    # 将输出逆归一化
    output1 = output1*1.0528-0.0128
    return output1


#创建云图数据
if __name__ == '__main__':
    #x,y的起始值
    valuex = 1.0
    valuey = -3.5
    #建立x与x/d(用于输出z)坐标数组
    inputx = []
    x_d = []
    i = 0
    j = 0
    while i <= 580:
        while j <= 140:
            inputx.append(valuex)
            j += 1
        j = 0
        x_d.append(valuex)
        valuex = valuex + 0.05
        valuex = round(valuex, 3)
        i += 1
    # print(x_d)#共701x2801=1963501个点

    # 建立y坐标数组
    inputy = []
    p = 0
    q = 0
    while p <= 580:
        while q <= 140:
            inputy.append(valuey)
            valuey = valuey + 0.05
            valuey = round(valuey, 3)
            q += 1
        q = 0
        valuey = -3.5
        p += 1
    # print(inputy[700])  # 共701x2801个点

    #建立y/d(用于输出z)坐标数组
    y_d = []
    value_y = -3.5
    nn = 0
    while nn <= 140:
        y_d.append(value_y)
        value_y = value_y + 0.05
        value_y = round(value_y, 3)
        nn += 1
    # print(len(y_d))

    # 建立z坐标数组
    CE = []#cooling efficiency
    x1 = 0
    y1 = 0
    # o = 2
    ave_CE = []
    while x1 <= 580:
        while y1 <= 140:
            output = predict(x_d[x1], y_d[y1])
            CE.append(output)
            y1 += 1
        y1 = 0
        x1 += 1
    # print(len(CE))
    print(np.sum(CE)/len(CE))


    #输出csv文件,在tecplot中处理
    with open('output_zhijie_16.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for x, y, ce in zip(inputx, inputy, CE):
            writer.writerow([x, y, ce])

















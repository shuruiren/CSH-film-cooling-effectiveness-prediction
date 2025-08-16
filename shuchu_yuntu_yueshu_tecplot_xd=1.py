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
a1 = [0, 0.5, 0.2321, 0.045, 0.1933]
#返回预测值
def predict (n):
    newnet = torch.load(".\\NEW_收敛孔_二维_展向约束_从xd1开始.pkl").cpu()
    x_d = (n-1) / 29
    input1 = torch.Tensor(np.array([a1[0], a1[1], a1[2], a1[3], a1[4], x_d])) #归一化后的数据
    out1 = newnet(input1)
    output1 = out1.detach().numpy()
    # print(output1)
    # 将输出逆归一化
    output1[0] = output1[0] * 0.75 - 1.86
    output1[1] = output1[1] * 1.146 + 0.179
    output1[2] = output1[2] * 0.687 + 0.104
    output1[3] = output1[3] * 2.146 + 0.697
    output1[4] = output1[4] * 1.877 + 0.221
    return output1

# def transpose_matrix(inputx):
#     return inputx.T.tolist()

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
    o = 2
    ave_CE = []
    while x1 <= 580:
        output = predict(x_d[x1])
        while y1 <= 140:
            y = y_d[y1]
            CE1 = output[2]/output[1]/1.253314 * math.exp(-2 * (y - output[0]) * (y - output[0]) / (output[1] * output[1]))
            CE2 = output[2]/output[1]/1.253314 * math.exp(-2 * (y + output[0]) * (y + output[0]) / (output[1] * output[1]))
            CE3 = output[4]/output[3]/1.253314 * math.exp(-2 * y * y / (output[3] * output[3]))
            CE.append(CE1 + CE2 + CE3)
            y1 += 1
        y1 = 0
        x1 += 1
    # print(len(CE))
    print(np.sum(CE)/len(CE))


    #输出csv文件,在tecplot中处理
    with open('output_YOUHUA_M15.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for x, y, ce in zip(inputx, inputy, CE):
            writer.writerow([x, y, ce])


    # xi = np.linspace(2, 30, 2801)
    # yi = np.linspace(-3.5, 3.5, 701)
    # X, Y = np.meshgrid(xi, yi)
    # Xiaolv = interpolate.griddata((inputx, inputy), CE, (X, Y), method='linear', fill_value=0)
    # fig, ax = plt.subplots(figsize=(14, 3.5))
    # jet = cm.get_cmap('jet')
    # # levels = range(0, 1, 0.1)
    # # cset1 = ax.contourf(X, Y, Xiaolv, levels, cmap=cm.jet)
    # cset1 = ax.contourf(X, Y, Xiaolv, cmap=jet)
    # ax.set_xlim(2, 30)
    # ax.set_ylim(-3.5, 3.5)
    # ax.set_xlabel("$x/d$", size=25)
    # ax.set_ylabel("$z/d$", size=25)
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    # cbar = fig.colorbar(cset1)
    # # cbar.set_label('strain(με)', size=18)
    # cbar.set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    # fig.savefig("预测结果.tif", bbox_inches='tight', dpi=1000, pad_inches=0.1)
    # # fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # 调整子图布局
    # # fig.savefig(figName + ".png", bbox_inches='tight', dpi=150, pad_inches=0.1)
    # plt.show()

    # # 云图显示
    # plt.rcParams['xtick.direction'] = 'in'
    # plt.rcParams['ytick.direction'] = 'in'
    # jet = cm.get_cmap('jet')
    # fig1, ax1 = plt.subplots(figsize=(14, 3.5))
    # cset1 = ax1.scatter(inputx, inputy, c=CE, cmap=jet, vmin=0, vmax=1, s=0.5, marker='s')
    # ax1.set_xlim(2, 30)
    # ax1.set_ylim(-3.5, 3.5)
    # ax1.set_xlabel("$x/d$", size=20)
    # ax1.set_ylabel("$z/d$", size=20)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # # cbar1 = fig1.colorbar(cset1)
    # # cbar1.set_label('E', size=18)
    # # cbar1.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # fig1.savefig("070pred.tif", bbox_inches='tight', dpi=1000, pad_inches=0.1)
    # plt.show()














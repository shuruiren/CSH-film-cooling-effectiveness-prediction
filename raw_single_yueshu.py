import torch.nn as nn
import torchsummary
class BPGNN(nn.Module):#定义了一个名为 BPGNN 的类，继承nn.Module类，是 nn.Module 的子类
    def __init__(self):#定义含有一个参数的构造（初始化）函数，self代表创建对象本身，python中构造函数的第一个参数必须是 `self`
        super(BPGNN, self).__init__() #并行调用父类的（初始化）方法：https://blog.csdn.net/weixin_44878336/article/details/124658574
        self.hidden1 = nn.Sequential(nn.Linear(in_features=6,out_features=50,bias=True),nn.PReLU())#,nn.ReLU()，in_features为输入的神经元个数，out_features为输出的神经元个数，bias=True为是否包含偏置
        self.hidden2 = nn.Sequential(nn.Linear(50, 100),nn.PReLU()) #self.hidden为BPGNN类的实例变量
        self.hidden3 = nn.Sequential(nn.Linear(100, 60),nn.PReLU())
        self.hidden4 = nn.Sequential(nn.Linear(60, 30),nn.PReLU())
        self.hidden5 = nn.Sequential(nn.Linear(30, 30), nn.PReLU())
        self.hidden_out = nn.Sequential(nn.Linear(30, 5))#, nn.Softmax(dim=-2)

    def forward(self, x): #前向传播方法
        out = self.hidden1(x)
        out = self.hidden2(out)
        out = self.hidden3(out)
        out = self.hidden4(out)
        out = self.hidden5(out)
        out = self.hidden_out(out)
        return out
# NET=BPGNN().cuda()
# torchsummary.summary(NET,input_size=[(1,1,5)])
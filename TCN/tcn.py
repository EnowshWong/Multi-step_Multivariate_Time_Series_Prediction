import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

#定义实现因果卷积的类，集成自torch.nn
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    #通过增加Padding的方式对卷积后的张量做切片实现因果卷积
    #tensor.contiguous()返回具有连续内存相同的张量
    #x为一般一维空洞卷积后的结果，第一维是批大小，第二维是通道数，第三维是序列长度，删除了卷积倒数padding个值，相当于
    #卷积向左移动了padding个值，实现了因果卷积
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

#定义了残差模块，即两个一维卷积的恒等映射
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        #定义第一个空洞卷积层
        #参数分别为，输入序列，输出序列，卷积核大小，滑动步长，填充0的个数，扩展系数
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        #根据padding大小对第一个空洞卷积层实现因果卷积，初始化padding
        self.chomp1 = Chomp1d(padding)
        #添加Relu激活函数与正则化方法
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        #堆叠同样的第二个空洞卷积层
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        #将卷积模块的所有组件，通过Sequential方法堆叠在一起
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        #padding保证了输入序列和输出序列相同，但卷积前的通道数和卷积后的通道数不一定相同
        #如果通道数不同，需要对x做一个逐元素的一维卷积使得与前面卷积的维度的相同
        #kernel_size = 1,做一维卷积
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        #以均值为0，标准差为0.01正态分布初始化，网络参数均为float32
        #torch中变量类型之间不能自动转换，因此网络参数类型和数据类型必须对应
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    #结合上述的卷积恒等映射，结合残差模块，投入relu函数完成残差模块的构建
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

#定义时间卷积网络架构
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        #num_channels保存了所有卷积层的通道数，它的长度就是卷积层的个数
        num_levels = len(num_channels)
        #空洞卷积以2的指数级扩张，以在不丢弃任何输入元素的前提下，扩大感受野
        for i in range(num_levels):
            #扩张系数变化
            dilation_size = 2 ** i
            #输入通道数
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            #输出通道数
            out_channels = num_channels[i]
            #一个残差模块的定义
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        #将所有残差模块组合起来形成深度时间卷积网络
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

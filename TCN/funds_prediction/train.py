import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
from TCN.funds_prediction.model import TCN
from TCN.funds_prediction.utils import *
import pandas as pd
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='Sequence Modeling - The Adding Problem')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_false',default=False,
                    help='use CUDA (default: True)')
#dropout:在残差网络的过滤器权重标准化之后，进行dropout操作
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (default: 0.0)')
#解决梯度爆炸现象，通过设定一阈值，将梯度限定在某范围内
parser.add_argument('--clip', type=float, default=0.15,
                    help='gradient clip, -1 means no clip (default: -1)')
#训练轮数
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit (default: 10)')
#1D卷积大小，关键参数，影响网络的感受野大小
parser.add_argument('--ksize', type=int, default=5,
                    help='kernel size (default: 7)')
#网络的层数
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
#序列长度
parser.add_argument('--seq_len', type=int, default=31,
                    help='sequence length (default: 400)')
#日志输出间隔
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='report interval (default: 100')
#学习率
# 需要
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 4e-3)')
#优化器
# 需要
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
#隐藏层神经元个数
# 需要
parser.add_argument('--nhid', type=int, default=40,
                    help='number of hidden units per layer (default: 30)')
#随机种子
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
args = parser.parse_args()

torch.manual_seed(args.seed)

batch_size = args.batch_size
seq_length = args.seq_len
epochs = args.epochs
lr = args.lr

# generate data
print("Producing data...")
x_train, y_train, x_test, y_test ,y_mean,y_std= generate_data(seq_length)
train_x,train_y = generate_train_samples(x_train,y_train,seq_length)
test_x,test_y = generate_test_samples(x_test,y_test,seq_length)

#输入维度和输出维度
input_channels = x_train.shape[1]
n_classes = y_train.shape[1]

# Note: We use a very simple setting here (assuming all levels have the same # of channels.
# 网络层数
channel_sizes = [args.nhid]*args.levels
kernel_size = args.ksize
dropout = args.dropout

# 创建TCN网络
model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

def train(epoch,epochs):
    global lr
    model.train()
    batch_idx = 1
    total_loss = 0
    train_output = torch.Tensor()
    for i in range(0, train_x.size()[0], batch_size):
        # 按批训练
        if i + batch_size > train_x.size()[0]:
            #最后一批不足batch_size数量的样本
            x, y = train_x[i:], train_y[i:]
        else:
            x, y = train_x[i:(i+batch_size)], train_y[i:(i+batch_size)]
        # 初始化梯度为0
        optimizer.zero_grad()
        # 得到模型的输出，模型输出为(-1,n_classes)
        output = model(x)
        train_output = torch.cat([train_output,output],0)
        # 使用均方误差损失,train_y的最后一个值为输入样本对应值
        loss = F.mse_loss(output, y[:,:,-1])
        # 反馈误差
        loss.backward()
        if args.clip > 0:
            # 若需要梯度稳定，可以采用clip策略
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        batch_idx += 1
        # item方法，获取loss的数值
        total_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            cur_loss = total_loss / args.log_interval
            processed = min(i+batch_size, train_x.size()[0])
            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                epoch, processed, train_x.size()[0], 100.*processed/train_x.size()[0], lr, cur_loss))
            total_loss = 0
    # 保存模型
    torch.save(model.state_dict(), 'params.pkl')
    return train_output

def evaluate():
    # 加载模型
    model.load_state_dict(torch.load('params.pkl'))
    preds_output = []
    # 将模型置为evaluate模式
    model.eval()
    output = model(test_x)
    test_loss = F.mse_loss(output,test_y[:,:,-1])
    test_acc = (torch.mean((test_y[:,0,-1] - output[:,0]) / test_y[:,0,-1]) + torch.mean((test_y[:,1,-1] - output[:,1]) / test_y[:,1,-1]))/2
    print('\nEpoch: {} Test set: Average loss: {:.6f}'.format(ep, test_loss.item()),' Accuracy: {:.6f}'.format(test_acc.item()))
    return test_loss,output

test_loss = []

for ep in range(1, epochs+1):
    train_output = train(ep,epochs)
    tloss , test_output= evaluate()
    test_loss.append(tloss)

# 数据可视化
train_output = train_output.detach().numpy()
test_output = test_output.detach().numpy()

# 数据反标准化还原
for i in range(y_train.shape[1]):
    train_output[:,i] = train_output[:,i] * y_std[i] + y_mean[i]
    test_output[:,i] = test_output[:,i] * y_std[i] + y_mean[i]
    y_train[:,i] = y_train[:,i] * y_std[i] + y_mean[i]
    y_test[:,i] = y_test[:,i] * y_std[i] + y_mean[i]

# 评价指标
mean_relative_error = 0.45 * np.mean(np.abs((test_output[:,0] - y_test[-seq_length:,0])) / y_test[-seq_length:,0]) + \
    0.55 * np.mean(np.abs((test_output[:,1] - y_test[-seq_length:,1])) / y_test[-seq_length:,1])
mean_square_error = 0.45 * np.mean(np.power(test_output[:,0] - y_test[-seq_length:,0],2)) + \
    0.55 * np.mean(np.power(test_output[:,1] - y_test[-seq_length:,1] / y_test[-seq_length:,1],2))
root_mean_square_error = 0.45 * np.sqrt(np.mean(np.power(test_output[:,0] - y_test[-seq_length:,0],2))) + \
    0.55 * np.sqrt(np.mean(np.power(test_output[:,1] - y_test[-seq_length:,1] / y_test[-seq_length:,1],2)))
mean_square_relative_error = 0.45 * np.mean(np.power((test_output[:,0] - y_test[-seq_length:,0]) / y_test[-seq_length:,0],2)) + \
    0.55 * np.mean(np.power((test_output[:,1] - y_test[-seq_length:,1]) / y_test[-seq_length:,1],2))
root_mean_square_relative_error = 0.45 * np.sqrt(np.mean(np.power((test_output[:,0] - y_test[-seq_length:,0]) / y_test[-seq_length:,0],2))) + \
    0.55 * np.sqrt(np.mean(np.power((test_output[:,1] - y_test[-seq_length:,1]) / y_test[-seq_length:,1],2)))
print('MAPE:',mean_relative_error)
print('MSE:',mean_square_error)
print('RMSE:',root_mean_square_error)
print('MSRE:',mean_square_relative_error)
print('RMSRE:',root_mean_square_relative_error)

# with open('results.csv','w') as f:
#     f.writelines([str(mean_relative_error) + ' ',str(mean_square_error) + ' ',str(root_mean_square_error)+' ',str(root_mean_square_relative_error)+' '])


plt.subplot(311)
plt.plot(y_train[:,0],color = 'black',label = 'train actual')
plt.plot(range(2 * seq_length - 1,2 * seq_length - 1 + train_output.shape[0]),train_output[:,0],color = 'yellow',label = 'train output')
plt.plot(range(y_train.shape[0],y_train.shape[0] + test_output.shape[0]),test_output[:,0],color = 'orange',label = 'test output')
plt.plot(range(y_train.shape[0],y_train.shape[0] + test_output.shape[0]),y_test[-seq_length:,0],color = 'green',label = 'test actual')
plt.title('fitting result of purchase')
plt.legend()
plt.tight_layout()

plt.subplot(312)
plt.plot(y_train[:,1],color = 'black',label = 'train actual')
plt.plot(range(2 * seq_length - 1,2 * seq_length - 1 + train_output.shape[0]),train_output[:,1],color = 'yellow',label = 'train output')
plt.plot(range(y_train.shape[0],y_train.shape[0] + test_output.shape[0]),test_output[:,1],color = 'orange',label = 'test output')
plt.plot(range(y_train.shape[0],y_train.shape[0] + test_output.shape[0]),y_test[-seq_length:,1],color = 'green',label = 'test actual')
plt.title('fitting result of redeem')
plt.legend()
plt.tight_layout()

plt.subplot(313)
plt.plot(test_loss,label = 'test loss')
plt.title('loss')
plt.legend()
plt.tight_layout()




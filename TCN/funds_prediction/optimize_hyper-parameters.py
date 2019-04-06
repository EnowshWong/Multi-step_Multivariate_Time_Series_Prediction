import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
from TCN.funds_prediction.model import TCN
from TCN.funds_prediction.utils import *
import optuna

parser = argparse.ArgumentParser(description='Sequence Modeling - The Adding Problem')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_false', default=False,
                    help='use CUDA (default: True)')
# dropout:在残差网络的过滤器权重标准化之后，进行dropout操作
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.0)')
# 解决梯度爆炸现象，通过设定一阈值，将梯度限定在某范围内
parser.add_argument('--clip', type=float, default=0.0,
                    help='gradient clip, -1 means no clip (default: -1)')
# 训练轮数
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit (default: 10)')
# 1D卷积大小，关键参数，影响网络的感受野大小
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
# 网络的层数
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
# 序列长度
parser.add_argument('--seq_len', type=int, default=31,
                    help='sequence length (default: 400)')
# 日志输出间隔
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='report interval (default: 100')
# 学习率
parser.add_argument('--lr', type=float, default=4e-3,
                    help='initial learning rate (default: 4e-3)')
# 优化器
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
# 隐藏层神经元个数
parser.add_argument('--nhid', type=int, default=40,
                    help='number of hidden units per layer (default: 30)')
# 随机种子
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
args = parser.parse_args()
# 7个参数，进行优化

torch.manual_seed(args.seed)

batch_size = args.batch_size
epochs = args.epochs
seq_length = args.seq_len

print("Producing data...")
dropout = args.dropout


def train(epoch, model, optimizer, train_x, train_y):
    global lr
    model.train()
    batch_idx = 1
    total_loss = 0
    train_output = torch.Tensor()
    for i in range(0, train_x.size()[0], batch_size):
        # 按批训练
        if i + batch_size > train_x.size()[0]:
            # 最后一批不足batch_size数量的样本
            x, y = train_x[i:], train_y[i:]
        else:
            x, y = train_x[i:(i + batch_size)], train_y[i:(i + batch_size)]
        # 初始化梯度为0
        optimizer.zero_grad()
        # 得到模型的输出，模型输出为(-1,n_classes)
        output = model(x)
        train_output = torch.cat([train_output, output], 0)
        # 使用均方误差损失,train_y的最后一个值为输入样本对应值
        loss = F.mse_loss(output, y[:, :, -1])
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
            processed = min(i + batch_size, train_x.size()[0])
            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                epoch, processed, train_x.size()[0], 100. * processed / train_x.size()[0], lr, cur_loss))
            total_loss = 0
    # 保存模型
    torch.save(model.state_dict(), 'params.pkl')
    return train_output


def evaluate(model, test_x, test_y):
    # 加载模型
    # model.load_state_dict(torch.load('params.pkl'))

    # 将模型置为evaluate模式
    model.eval()
    # 得到测试结果
    output = model(test_x)

    test_loss = F.mse_loss(output, test_y[:, :, -1])
    test_acc = (torch.mean((test_y[:, 0, -1] - output[:, 0]) / test_y[:, 0, -1]) + torch.mean(
        (test_y[:, 1, -1] - output[:, 1]) / test_y[:, 1, -1])) / 2
    print('\nTest set: Average loss: {:.6f}'.format(test_loss.item()), ' Accuracy: {:.6f}'.format(test_acc.item()))
    return test_loss.item(), output


def prediction(model, predict_x):
    # model.load_state_dict(torch.load('params.pkl'))
    model.eval()
    # 预测未来30天
    output = model(predict_x)
    return output


def start_train_and_test(epochs, model, optimizer, train_x, train_y, test_x, test_y):
    test_loss: list = []
    for ep in range(1, epochs + 1):
        train_output = train(ep, model, optimizer, train_x, train_y)
        tloss, test_output = evaluate(model, test_x, test_y)
        test_loss.append(tloss)
    return np.mean(np.array(test_loss))


def objective(trail: optuna.Trial):
    print("Produce data...")
    x_train, y_train, x_test, y_test, y_mean, y_std = generate_data(seq_length)
    train_x, train_y = generate_train_samples(x_train, y_train, seq_length)
    test_x, test_y = generate_test_samples(x_test, y_test, seq_length)
    # 输入维度和输出维度
    input_channels = x_train.shape[1]
    n_classes = y_train.shape[1]

    # 网络层数
    level = trail.suggest_int('levels', 5, 10)
    # 每层的神经元个数
    nhid = trail.suggest_int('nhid',30,50)
    channel_sizes = [nhid] * level
    # 学习率
    lr = trail.suggest_loguniform('lr', 1e-5, 1e-1)
    # 卷积核大小
    kernel_size = trail.suggest_int('kernel_size', 3, 10)

    model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)
    #
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
    loss = start_train_and_test(epochs, model, optimizer, train_x, train_y, test_x, test_y)
    return loss


def main():
    study = optuna.create_study()
    study.optimize(objective, n_trials=25)
    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))


if __name__ == '__main__':
    main()


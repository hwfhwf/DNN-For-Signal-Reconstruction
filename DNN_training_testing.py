import torch
import numpy as np
import torch.utils.data as Data
from torch import nn, optim
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
import scipy.io as sio  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

##数据准备(matlab导入)
data=sio.loadmat('matlab.mat')
data1=sio.loadmat('testset.mat')
LSB=2/2**1
x_IF=data['Xx']
testset=data1['Xx']
testset=np.concatenate([np.real(testset),np.imag(testset)],0)
testset_sampled=np.round((testset)/LSB)*LSB
testset=testset.T
testset_sampled=torch.Tensor(testset_sampled.T).view(-1,1,1,64).to(device)
x_IF=np.concatenate([np.real(x_IF),np.imag(x_IF)],0)
x_IF_sampled=np.round((x_IF)/LSB)*LSB
loss_init=((x_IF-x_IF_sampled)**2).mean()*1e4
num_data=x_IF_sampled.shape[1]
num_antenna=x_IF_sampled.shape[0]

#torch
x_IF=torch.Tensor(x_IF.T)
x_IF_sampled=torch.Tensor(x_IF_sampled.T).view(num_data,1,1,num_antenna)#转换
# x_IF_sampled+=torch.normal(0,0.5,x_IF_sampled.shape)
batch_size=256
dataset = Data.TensorDataset(x_IF_sampled, x_IF) #分清训练数据和目标维度
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)


class Residual(nn.Module):  # 本类已保存在d2lzh_pytorch包中方便以后使用
    def __init__(self, in_channels, out_channels):
        super(Residual, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.fc1(X)))
        Y = self.bn2(self.fc2(Y))
        return F.relu(Y+X)
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(64, 1024),
            nn.ReLU(),
            Residual(1024,1024),
            Residual(1024,1024),
            Residual(1024,1024),
            Residual(1024,1024),
            nn.Linear(1024, 64) 
        )

    def forward(self, img):
        output = self.fc(img.view(img.shape[0], -1))
        return output

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (1,3),1,(0,1)),
            nn.ReLU(),
            nn.Conv2d(32, 32,  (1,3),1,(0,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32,  (1,3),1,(0,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32,  (1,3),1,(0,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32,  (1,3),1,(0,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32,  (1,3),1,(0,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32,  (1,3),1,(0,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32,  (1,3),1,(0,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32,  (1,3),1,(0,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32,  (1,3),1,(0,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1,  (1,3),1,(0,1)),
        )
    def forward(self, img):
        feature = self.conv(img)
        output=feature.view(img.shape[0],-1)
        return output

def train_ch5(net, train_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)#cpu&gpu
    plt.ion()
    print("training on ", device)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        # optim.lr_scheduler.StepLR(optimizer, 100 , gamma=0.3, last_epoch=epoch)
        print('lr= ',optimizer.param_groups[0]['lr'])
        for X, y in train_iter:
            X = X.to(device)#cpu&gpu
            y = y.to(device)
            y_hat = net(X)
            l=torch.nn.MSELoss()(y_hat,y)
            #l = ((y_hat-y)**2).mean()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item() #cpu
            n += y.shape[0]
            batch_count += 1
        plt.scatter(epoch+1,train_l_sum*1e4 / batch_count)
        plt.draw()
        plt.pause(0.01)
        test_trained=net(testset_sampled).cpu().detach().numpy()
        testerror=((test_trained-testset)**2).mean()*1e4
        print('|epoch %d|,|train loss  %.4f|  |test loss1  %.4f|  time %.1f sec'
              % (epoch + 1, train_l_sum*1e4 / batch_count,  testerror,  time.time() - start))
    plt.ioff()
net = ResNet()
# net=ConvNet()
# net=torch.load('training64angleIQdeep4.pkl')
# for name, param in net.named_parameters():检验框架
#     print(param.dtype)
#     # print(name, param.size())

lr, num_epochs = 1e-3 ,5
optimizer = torch.optim.Adam([{'params': net.parameters(), 'initial_lr': 0.001}], lr=lr)
train_ch5(net, data_iter, batch_size, optimizer, device, num_epochs)
#save data
X=net(x_IF_sampled).detach().numpy()
for k in range(100):
    print(((X[k]-x_IF[k].numpy())**2).mean()*1e4)
sio.savemat('Xx_round.mat', mdict={'Xx_round':X})

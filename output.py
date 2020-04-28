import torch
import numpy as np
from torch import nn, optim
import scipy.io as sio  
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
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
        return F.relu(Y + X)
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
            nn.Conv2d(1, 64, (1,3),1,(0,1)),
            nn.ReLU(),
            nn.Conv2d(64, 64,  (1,3),1,(0,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64,  (1,3),1,(0,1)),       
            nn.BatchNorm2d(64),           
            nn.ReLU(),
            nn.Conv2d(64, 64,  (1,3),1,(0,1)),       
            nn.BatchNorm2d(64),           
            nn.ReLU(),
            nn.Conv2d(64, 64,  (1,3),1,(0,1)),       
            nn.BatchNorm2d(64),           
            nn.ReLU(),
            nn.Conv2d(64, 64,  (1,3),1,(0,1)),       
            nn.BatchNorm2d(64),           
            nn.ReLU(),
            nn.Conv2d(64, 64,  (1,3),1,(0,1)),       
            nn.BatchNorm2d(64),           
            nn.ReLU(),
            nn.Conv2d(64, 64,  (1,3),1,(0,1)),       
            nn.BatchNorm2d(64),           
            nn.ReLU(),
            nn.Conv2d(64, 64,  (1,3),1,(0,1)),       
            nn.BatchNorm2d(64),           
            nn.ReLU(),
            nn.Conv2d(64, 64,  (1,3),1,(0,1)),       
            nn.BatchNorm2d(64),           
            nn.ReLU(),
            nn.Conv2d(64, 1,  (1,3),1,(0,1)),
        )
    def forward(self, img):
        feature = self.conv(img)
        output=feature.view(img.shape[0],-1)
        return output

##数据准备(matlab导入)
data=sio.loadmat('test.mat') 
LSB=2/2**1
x_IF=data['Xx']
x_IF=np.concatenate([np.real(x_IF),np.imag(x_IF)],0)
x_IF_sampled=np.round((x_IF)/LSB)*LSB
num_data=x_IF_sampled.shape[1]
num_antenna=x_IF_sampled.shape[0]
x_IF=torch.Tensor(x_IF.T)
x_IF_sampled=torch.tensor(x_IF_sampled.T,dtype=torch.float32).view(num_data,1,1,num_antenna)#转换
net=torch.load('training64angleIQdeep4.pkl')
# net=torch.load('FP16.pkl')
# X=net(x_IF_sampled).detach().numpy()+x_IF_sampled.view(num_data,num_antenna).numpy()
X=net(x_IF_sampled).detach().numpy()
for k in range(4000,4030):     
    print(((X[k]-x_IF[k].numpy())**2).mean()*1e4)
sio.savemat('Xx_round.mat', mdict={'Xx_round':X})
##剪枝
# total=0
# for param in net.parameters():
#     if len(param.shape)==2:
#         for i in range(param.shape[0]):
#             for j in range(param.shape[1]):
#                 if abs(param[i][j])<=0.01:
#                     param[i][j]=0
#                     total+=1
#     else:
#         for i in range(param.shape[0]):
#             if abs(param[i])<=0.01:
#                 param[i]=0
#                 total+=1
# print(total)
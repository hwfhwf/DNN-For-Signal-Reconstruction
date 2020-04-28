# [DNN]
# [Deep Neural Network-Based Quantizied Signal Reconsruction for DOA Estimation]

## Training and Testing
  - [DNN_training_testing]

## Output Data for DOA Estimation
  - [Reconstruct Signal (.mat file for MATLAB)]

## Workflow for Reconstruction and performance test
### 1.Data Initialization
[DNN_training_testing]
```python
data=sio.loadmat('matlab.mat')         # load training data, locate dataset in /dataset
data1=sio.loadmat('testset.mat')       # load test data

LSB=2/2**1                             # LSB is resolution of B-bit ADC
x_IF=data['Xx']                        # training data
testset=data1['Xx']                    # testing data

testset=np.concatenate([np.real(testset),np.imag(testset)],0)     #seperate complex signals
testset_sampled=np.round((testset)/LSB)*LSB                       #quantize the signal
testset=testset.T
testset_sampled=torch.Tensor(testset_sampled.T).view(-1,1,1,64).to(device)

x_IF=np.concatenate([np.real(x_IF),np.imag(x_IF)],0)
x_IF_sampled=np.round((x_IF)/LSB)*LSB

loss_init=((x_IF-x_IF_sampled)**2).mean()*1e4
num_data=x_IF_sampled.shape[1]                                   #signals' number
num_antenna=x_IF_sampled.shape[0]                                #number of antennas

x_IF=torch.Tensor(x_IF.T)                                        #transform numpy data to tensor
x_IF_sampled=torch.Tensor(x_IF_sampled.T).view(num_data,1,1,num_antenna)
batch_size=256
dataset = Data.TensorDataset(x_IF_sampled, x_IF) 
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)   #data have prepared
```
### 2.Form a Network
```python
class Residual(nn.Module):                       #Residule Block
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
        
class ResNet(nn.Module):                       #DNN with Residule Block
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
```
### 3.Save Trained Network
```python
torch.save(net,'MyDNN.pkl')
```
[Network parameters (.pkl file)]
### 4.Output Reconstructed Signal
[Reconstruct Signal (.mat file for MATLAB)]
```python
data=sio.loadmat('test.mat')                              #load test sinal
LSB=2/2**1
x_IF=data['Xx']
x_IF=np.concatenate([np.real(x_IF),np.imag(x_IF)],0)
x_IF_sampled=np.round((x_IF)/LSB)*LSB
num_data=x_IF_sampled.shape[1]
num_antenna=x_IF_sampled.shape[0]
x_IF=torch.Tensor(x_IF.T)
x_IF_sampled=torch.tensor(x_IF_sampled.T,dtype=torch.float32).view(num_data,1,1,num_antenna)
net=torch.load('MyDNN.pkl')
X=net(x_IF_sampled).detach().numpy()
sio.savemat('Xx_round.mat', mdict={'Xx_round':X})       #put reconstructed signal in .mat file
```
### 4.DOA Estimation via MUSIC Algorithm
#### look up for source code in ../DOA-via-MATLAB/[music.m]

# APPENDIX : Files Instructions
#### [DNN_training_testing.py] : Train and test the dataset
#### [output.py] : Generate reconstruncted signal
#### [MyDNN.pkl] : Network parameters
#### DOA-via-MATLAB/[music.m] : Input reconstruncted signal to estimate DOA
#### DOA-via-MATLAB/[creat_data.m] : Generate all data including [matlab.mat] , [testset.mat] , [test.mat] , [Xx_round.mat]
#### DOA-via-MATLAB/[creat_figure.m] : Plot figures in paper
#### dataset/[matlab.mat] : Training data
#### dataset/[testset.mat] : Test data
#### dataset/[test.mat] : Dataset for signal recovery
#### dataset/[Xx_round.mat] : Reconstructed signal


[DNN_training_testing.py]:<https://github.com/hwfhwf/DNN-For-Signal-Reconstrction/blob/master/DNN_training_testing.py>
[output.py]:<https://github.com/hwfhwf/DNN-For-Signal-Reconstrction/blob/master/output.py>
[MyDNN.pkl]:<https://github.com/hwfhwf/DNN-For-Signal-Reconstrction/blob/master/MyDNN.pkl>
[music.m]:<https://github.com/hwfhwf/DNN-For-Signal-Reconstrction/blob/master/DOA-via-MATLAB/music.m>
[creat_data.m]:<https://github.com/hwfhwf/DNN-For-Signal-Reconstrction/blob/master/DOA-via-MATLAB/creat_data.m>
[creat_figure.m]:<https://github.com/hwfhwf/DNN-For-Signal-Reconstrction/blob/master/DOA-via-MATLAB/creat_figure.m>
[matlab.mat]:<https://github.com/hwfhwf/DNN-For-Signal-Reconstrction/blob/master/dataset/matlab.mat>
[testset.mat]:<https://github.com/hwfhwf/DNN-For-Signal-Reconstrction/blob/master/dataset/testset.mat>
[test.mat]:<https://github.com/hwfhwf/DNN-For-Signal-Reconstrction/blob/master/dataset/test.mat>
[Xx_round.mat]:<https://github.com/hwfhwf/DNN-For-Signal-Reconstrction/blob/master/dataset/Xx_round.mat>   
[Network parameters (.pkl file)]:<https://github.com/hwfhwf/DNN-For-Signal-Reconstrction/blob/master/output.py>
[Reconstruct Signal (.mat file for MATLAB)]:<https://github.com/hwfhwf/DNN-For-Signal-Reconstrction/blob/master/output.py>   
[DNN_training_testing]: <https://github.com/hwfhwf/DNN-For-Signal-Reconstrction/blob/master/DNN_training_testing.py>
[DNN]: <https://github.com/hwfhwf/DNN-For-Signal-Reconstrction>
[Deep Neural Network-Based Quantizied Signal Reconsruction for DOA Estimation]: <论文网址>
 

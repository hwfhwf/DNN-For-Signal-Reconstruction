
clear; close all; 
clc
% format lg;
sigNum =2;
theta=deg2rad([-10;5])
vec = @(MAT) MAT(:);
vecH = @(MAT) MAT(:).';
M=32;%%阵元个数
S0=[1; zeros(M-1, 1)];
d = 0.5; 
steerVec = @(angTmp) exp(1j*2*pi*[0:1:M-1].'*d*sin(vecH(angTmp)));
m=0:M-1;
N=1000 %%%%%%%快拍数
As=steerVec(theta(1:end));
SNR=30;  % 信噪比（可变）
p=10.^(SNR/10); 
signal = sqrt(p/2) * (-1+2*rand(N, sigNum)) + sqrt(p/2)*j* (-1+2*rand(N, sigNum));
Xs = As * signal.';
Xn =(sqrt(1/2) *randn(N, M) + sqrt(1/2)* j* randn(N, M)).';
Xx=Xn+Xs;
max_value=max(max(max(abs(real(Xx)))),max(max(abs(real(Xx)))))
Xx=Xx/max_value
LSB=2/2.^1
Xx_round=round(Xx/LSB)*LSB
xxl=deg2rad([-39.99:0.01:40].');
vector=zeros(M,1)
vector_round=zeros(M,1)
for k = round(unifrnd (300, 800, 1, 1))
vector=vector+Xx(:,k)
vector_round=vector_round+Xx_round(:,k)
end
s1 = abs(vector'*steerVec(xxl));
s2 = abs(vector_round'*steerVec(xxl));
[pks1,locs1] = findpeaks(s1,'minpeakdistance',100)
[pks2,locs2] = findpeaks(s2,'minpeakdistance',100)
error1=(min(abs(locs1-3000))/100)+(min(abs(locs1-4500))/100)%+(min(abs(locs1-5000))/100)
error2=(min(abs(locs2-3000))/100)+(min(abs(locs2-4500))/100)%+(min(abs(locs2-5000))/100)
s1=s1/max(s1);
s2=s2/max(s2);
figure;
subplot(121)
plot(rad2deg(xxl),s1,'-r','linewidth',3)
xlabel('sita');
ylabel('Amplitude');
grid on
hold on;
stem(rad2deg(theta(1:end)), zeros(sigNum, 1), '-*', 'BaseValue',1);
subplot(122)
plot(rad2deg(xxl),s2,'-r','linewidth',3)
xlabel('sita');
ylabel('Amplitude');
grid on
hold on;
stem(rad2deg(theta(1:end)), zeros(sigNum, 1), '-*', 'BaseValue',1);

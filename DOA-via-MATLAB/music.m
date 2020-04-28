clear; close all; 
clc
vec = @(MAT) MAT(:);
vecH = @(MAT) MAT(:).';
M=32;%%ÕóÔª¸öÊý
S0=[1; zeros(M-1, 1)];
d = 0.5; 
steerVec = @(angTmp) exp(1j*2*pi*[0:1:M-1].'*d*sin(vecH(angTmp)));
m=0:M-1;
error=0;error_round=0;error_1bit=0;error_2bit=0;error_3bit=0;error_4bit=0;

num_sum=0;
load 'random_total.mat'
load 'Xx_round.mat'
load 'test.mat'
Xx_round=Xx_round.';
Xx_round=Xx_round(1:32,:)+1j*Xx_round(33:64,:);
LSB1=2/2;
LSB2=2/4;
LSB3=2/8;
LSB4=2/16;
Xx_1bit=round(Xx/LSB1)*LSB1;
Xx_2bit=round(Xx/LSB2)*LSB2;
Xx_3bit=round(Xx/LSB3)*LSB3;
Xx_4bit=round(Xx/LSB4)*LSB4;
xxl=deg2rad([-39.99:0.01:40].');
for epochs=3015
theta=deg2rad(random_total(:,epochs)); 
sigNum =length(theta);
error_tmp=0;error_round_tmp=0;error_1bit_tmp=0;error_2bit_tmp=0;error_3bit_tmp=0;error_4bit_tmp=0;
% if epochs>10
%     epoch=epochs-10*floor((epochs-1)/10)+40;
% else
%     epoch=epochs+40;
% end
vector=zeros(M,1);
vector_round=zeros(M,1);
vector_1bit=zeros(M,1);
vector_2bit=zeros(M,1);
vector_3bit=zeros(M,1);
vector_4bit=zeros(M,1);
R=zeros(M,M);
R_round=zeros(M,M);
R_1bit=zeros(M,M);
R_2bit=zeros(M,M);
R_3bit=zeros(M,M);
R_4bit=zeros(M,M);

for k =  (epochs-1)*5+1: epochs*5%round(unifrnd ((epoch-1)*100+1, epoch*100, 1, 50))
    
vector=Xx(:,k);
R=R+vector*vector';

vector_round=Xx_round(:,k);
R_round=R_round+vector_round*vector_round';

vector_1bit=Xx_1bit(:,k);
R_1bit=R_1bit+vector_1bit*vector_1bit';

vector_2bit=Xx_2bit(:,k);
R_2bit=R_2bit+vector_2bit*vector_2bit';

vector_3bit=Xx_3bit(:,k);
R_3bit=R_3bit+vector_3bit*vector_3bit';

vector_4bit=Xx_4bit(:,k);
R_4bit=R_4bit+vector_4bit*vector_4bit';
end


[V,D]=eig(R);
[D_sort,index] = sort(diag(D),'descend');
D_sort = D_sort(index);V_sort = V(:,index);
UN=V_sort(:,sigNum+1:M);

[V_round,D_round]=eig(R_round);
[D_round_sort,index] = sort(diag(D_round),'descend');
D_round_sort = D_round_sort(index);V_round_sort = V_round(:,index);
UN_round=V_round_sort(:,sigNum+1:M);

[V_1bit,D_1bit]=eig(R_1bit); 
[D_1bit_sort,index] = sort(diag(D_1bit),'descend');
D_1bit_sort = D_1bit_sort(index);V_1bit_sort = V_1bit(:,index);
UN_1bit=V_1bit_sort(:,sigNum+1:M);

[V_2bit,D_2bit]=eig(R_2bit); 
[D_2bit_sort,index] = sort(diag(D_2bit),'descend');
D_2bit_sort = D_2bit_sort(index);V_2bit_sort = V_2bit(:,index);
UN_2bit=V_2bit_sort(:,sigNum+1:M);

[V_3bit,D_3bit]=eig(R_3bit); 
[D_3bit_sort,index] = sort(diag(D_3bit),'descend');
D_3bit_sort = D_3bit_sort(index);V_3bit_sort = V_3bit(:,index);
UN_3bit=V_3bit_sort(:,sigNum+1:M);

[V_4bit,D_4bit]=eig(R_4bit); 
[D_4bit_sort,index] = sort(diag(D_4bit),'descend');
D_4bit_sort = D_4bit_sort(index);V_4bit_sort = V_4bit(:,index);
UN_4bit=V_4bit_sort(:,sigNum+1:M);
for iii=1:length(xxl)
a=steerVec(xxl(iii));
Pmusic(iii)=abs(1/(a'*(UN*UN')*a));
Pmusic_round(iii)=abs(1/(a'*(UN_round*UN_round')*a));
Pmusic_1bit(iii)=abs(1/(a'*(UN_1bit*UN_1bit')*a));
Pmusic_2bit(iii)=abs(1/(a'*(UN_2bit*UN_2bit')*a));
Pmusic_3bit(iii)=abs(1/(a'*(UN_3bit*UN_3bit')*a));
Pmusic_4bit(iii)=abs(1/(a'*(UN_4bit*UN_4bit')*a));
end

Pmusic=Pmusic/(max(Pmusic)+eps);
Pmusic_round=Pmusic_round/(max(Pmusic_round)+eps);
Pmusic_1bit=Pmusic_1bit/(max(Pmusic_1bit)+eps);
Pmusic_2bit=Pmusic_2bit/(max(Pmusic_2bit)+eps);
Pmusic_3bit=Pmusic_3bit/(max(Pmusic_3bit)+eps);
Pmusic_4bit=Pmusic_4bit/(max(Pmusic_4bit)+eps);

[pks1,locs1] = findpeaks(Pmusic);
[pks2,locs2] = findpeaks(Pmusic_round);
[pks3,locs3] = findpeaks(Pmusic_1bit);
[pks4,locs4] = findpeaks(Pmusic_3bit);
[pks5,locs5] = findpeaks(Pmusic_4bit);
[pks6,locs6] = findpeaks(Pmusic_2bit);
for i=1:sigNum
    error_tmp=error_tmp+(min(abs(locs1-4000-100*random_total(sigNum,epochs)))/100).^2;
    error_round_tmp=error_round_tmp+(min(abs(locs2-4000-100*random_total(sigNum,epochs)))/100).^2;
    error_1bit_tmp=error_1bit_tmp+(min(abs(locs3-4000-100*random_total(sigNum,epochs)))/100).^2;
    error_2bit_tmp=error_2bit_tmp+(min(abs(locs6-4000-100*random_total(sigNum,epochs)))/100).^2;
    error_3bit_tmp=error_3bit_tmp+(min(abs(locs4-4000-100*random_total(sigNum,epochs)))/100).^2;
    error_4bit_tmp=error_4bit_tmp+(min(abs(locs5-4000-100*random_total(sigNum,epochs)))/100).^2;
end
error_tmp=error_tmp/sigNum;error_round_tmp=error_round_tmp/sigNum;
error_1bit_tmp=error_1bit_tmp/sigNum;error_2bit_tmp=error_2bit_tmp/sigNum;
error_3bit_tmp=error_3bit_tmp/sigNum;error_4bit_tmp=error_4bit_tmp/sigNum;
num_sum=num_sum+1;
error=error+error_tmp;
error_round=error_round+error_round_tmp;
error_1bit=error_1bit+error_1bit_tmp;
error_2bit=error_2bit+error_2bit_tmp;
error_3bit=error_3bit+error_3bit_tmp;
error_4bit=error_4bit+error_4bit_tmp;
if rem(epochs,1000)==0
    a1(epochs/1000)=error_1bit;    a2(epochs/1000)=error_2bit; 
    a3(epochs/1000)=error_3bit;
    a4(epochs/1000)=error_4bit;    a5(epochs/1000)=error_round;
    error_1bit=0;error_2bit=0;error_3bit=0;error_4bit=0;error_round=0;
end
end
figure;
plot(rad2deg(xxl),Pmusic,'-b','linewidth',2)
hold on;
plot(rad2deg(xxl),Pmusic_round,'-r','linewidth',2)
plot(rad2deg(xxl),Pmusic_1bit,'-m','linewidth',2)
plot(rad2deg(xxl),Pmusic_2bit,'-g','linewidth',2)
plot(rad2deg(xxl),Pmusic_3bit,'-k','linewidth',2)
xlabel('sita');
ylabel('Amplitude');
stem(rad2deg(theta(1:end)), zeros(sigNum, 1), '--k', 'BaseValue',1,'linewidth',1);
legend('signal','training','1bit','2bit','3bit','truth ground')
grid on

figure;
plot(rad2deg(xxl),Pmusic_round,'-k','linewidth',1)
hold on;
plot(rad2deg(xxl),Pmusic_2bit,'-r','linewidth',1)
plot(rad2deg(xxl),Pmusic_3bit,'-g','linewidth',1)
xlabel('Spatial Angle(¡ã)','FontName','Times New Roman');
ylabel('Spectrum','FontName','Times New Roman');
stem(rad2deg(theta(1:end)), zeros(sigNum, 1), '--k', 'BaseValue',1,'linewidth',1);
set(gca,'FontSize',13)
h=legend('Recontrcted Signal','2bit Quantized Signal','3bit Quantized Signal','Truth Ground');
set(h,'FontName','Times New Roman','FontSize',13,'FontWeight','normal')
grid on
% %% 3个角度 100W训练集 1000次平均 相同信号 噪声不同
% 
% SNR=[10:10:50];
% error_1bit=[279,202,196,195,205]/1000;
% error_2bit=[221,139,123,143,130]/1000;
% error_3bit=[184,69.5,69.3,55,59]/1000;
% error_4bit=[139,41,25,23,19]/1000;
% error_proposed_DNN=[133,103,81,78,84]/1000;
% figure();
% semilogy(SNR,error_1bit,'o-','linewidth',2,'MarkerSize',8);
% xlabel('SNR(dB)','FontName','Times New Roman');
% ylabel('MSE','FontName','Times New Roman');
% grid on;hold on;
% semilogy(SNR,error_2bit,'x-','linewidth',2,'MarkerSize',10);
% semilogy(SNR,error_3bit,'*-','linewidth',2,'MarkerSize',10);
% semilogy(SNR,error_4bit,'s-','linewidth',2,'MarkerSize',10);
% semilogy(SNR,error_proposed_DNN,'.--b','linewidth',1.5,'MarkerSize',25);
% set(gca,'FontSize',13)
% h=legend('1bit','2bit','3bit','4bit','Proposed DNN');
% set(h,'FontName','Times New Roman','FontSize',13,'FontWeight','normal')

%% 3个角度 100W训练集 1000次平均 相同信号 噪声不同

SNR=[10:10:50];
error_1bit=[279,202,196,195,205]/1000;
error_2bit=[221,139,123,143,130]/1000;
error_3bit=[184,69.5,69.3,55,59]/1000;
error_4bit=[139,41,25,23,19]/1000;
error_32FP=[133,103,81,78,84]/1000;
error_16FP=[129,95,81.1,80.8,80.2]/1000;
error_Pruning=[164,118,104,95,98]/1000;
figure();
semilogy(SNR,error_1bit,'o-','linewidth',2,'MarkerSize',8);
xlabel('SNR(dB)','fontsize',15,'fontname','Times New Roman');
ylabel('MSE','fontsize',15,'fontname','Times New Roman');
grid on;hold on;
semilogy(SNR,error_2bit,'x-','linewidth',2,'MarkerSize',10);
semilogy(SNR,error_3bit,'.-','linewidth',2,'MarkerSize',25);
semilogy(SNR,error_4bit,'*-','linewidth',2,'MarkerSize',10);
semilogy(SNR,error_32FP,'.--b','linewidth',1.5,'MarkerSize',20);
semilogy(SNR,error_16FP,'^--r','linewidth',1.5,'MarkerSize',9);
semilogy(SNR,error_Pruning,'s--k','linewidth',1.5,'MarkerSize',10);
set(gca,'FontSize',13)
h=legend('1bit','2bit','3bit','4bit','32FP','16FP','Pruning');
set(h,'FontName','Times New Roman','FontSize',13,'FontWeight','normal')

%% 3个角度 100W训练集 Conv Res对比
% 
% dB=[10,20,30,40,50];
% error_1bit=[279,202,196,195,205]/1000;
% error_2bit=[221,139,123,143,130]/1000;
% error_3bit=[184,69.5,69.3,55,59]/1000;
% error_4bit=[139,41,25,23,19]/1000;
% error_ResNet=[205,118,122,117,110]/1000;
% error_ConvNet=[276,221,178,193,177]/1000;
% 
% figure();
% semilogy(dB,error_1bit,'.-r','linewidth',2,'MarkerSize',20);
% xlabel('SNR(dB)','FontName','Times New Roman');
% ylabel('MSE','FontName','Times New Roman');
% grid on;hold on;
% semilogy(dB,error_2bit,'.-b','linewidth',2,'MarkerSize',20);
% semilogy(dB,error_3bit,'.-k','linewidth',2,'MarkerSize',20);
% semilogy(dB,error_4bit,'.-g','linewidth',2,'MarkerSize',20);
% semilogy(dB,error_ResNet,'.--','linewidth',2,'MarkerSize',20);
% semilogy(dB,error_ConvNet,'.--','linewidth',2,'MarkerSize',20);
% set(gca,'FontSize',13)
% h=legend('1bit','2bit','3bit','4bit','Proposed DNN','DnCNN');
% set(h,'FontName','Times New Roman','FontSize',13,'FontWeight','normal')
% %% 3个角度 100W训练集 1000次平均
% %5 snapshot 200 trial
% % clear; close all; 
% % clc
% SNR=[10:10:50];
% error_2bit=a1/1000;
% error_3bit=a2/1000;
% error_4bit=a3/1000;
% error_ResNet=a4/1000;
% figure();
% semilogy(SNR,error_2bit,'.-','linewidth',2,'MarkerSize',20);
% xlabel('SNR(dB)');ylabel('MSE(dB)');
% grid on;hold on;
% semilogy(SNR,error_3bit,'.-','linewidth',2,'MarkerSize',20);
% semilogy(SNR,error_4bit,'.-','linewidth',2,'MarkerSize',20);
% semilogy(SNR,error_ResNet,'.-','linewidth',2,'MarkerSize',20);
% 
% legend('2bit','3bit','4bit','ResNet')
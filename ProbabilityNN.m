close all; 
clear all; 
clc;
%% load divided input data set
load fisheriris
% coding the classes
a = 1;
b = 2;
c = 3;
% define training inputs
rand_ind = randperm(50);
trainSeto = meas(rand_ind(1:35),:);
trainSeto=trainSeto';
trainVers = meas(50 + rand_ind(1:35),:);
trainVers=trainVers';
trainVirg = meas(100 + rand_ind(1:35),:);
trainVirg=trainVirg';
trainInp = [trainSeto trainVers trainVirg];
% define targets
tmp1 = repmat(a,1,length(trainSeto));
tmp2 = repmat(b,1,length(trainVers));
tmp3 = repmat(c,1,length(trainVirg));
T = [tmp1 tmp2 tmp3];
%% choose a spread constant (1st step)
spread = 1.1;
Cor = zeros(2,109);
Sp = zeros(1,109);
Sp(1,1) = spread;
for i = 1:109,
spread = spread - 0.01;
Sp(1,i) = spread;
% create a neural network
net = newpnn(trainInp,ind2vec(T),spread);
% simulate PNN on training data
Y = sim(net,trainInp);
% convert PNN outputs
Y = vec2ind(Y);
% define validation vector
rand_ind = randperm(50);
valSeto = meas(rand_ind(1:20),:);
valSeto= valSeto';
valVers = meas(50 + rand_ind(1:20),:);
valVers=valVers';
valVirg = meas(100 + rand_ind(1:20),:);
valVirg=valVirg';

valInp = [valSeto valVers valVirg];
tmp1 = repmat(a,1,length(valSeto));
tmp2 = repmat(b,1,length(valVers));
tmp3 = repmat(c,1,length(valVirg));
valT = [tmp1 tmp2 tmp3];
Yval = sim(net,valInp,[],[],ind2vec(valT));
Yval = vec2ind(Yval);
% calculate [%] of correct classifications
Cor(1,i) = 100 * length(find(T==Y)) / length(T);
Cor(2,i) = 100 * length(find(valT==Yval)) / length(valT);
end
figure
pl = plot(Sp,Cor);
set(pl,{'linewidth'},{1,3}');
%% choose a spread constant (2nd step)
spread = 0.25;
Cor1 = zeros(2,200);
Sp1 = zeros(1,200);
Sp1(1,1) = spread;
for i = 1:200,
spread = spread - 0.0001;
Sp1(1,i) = spread;
% create a neural network
net = newpnn(trainInp,ind2vec(T),spread);
% simulate PNN on training data
Y = sim(net,trainInp);
% convert PNN outputs
Y = vec2ind(Y);
Yval = sim(net,valInp,[],[],ind2vec(valT));
Yval = vec2ind(Yval);
% calculate [%] of correct classifications
Cor1(1,i) = 100 * length(find(T==Y)) / length(T);
Cor1(2,i) = 100 * length(find(valT==Yval)) / length(valT);
end
figure
pl1 = plot(Sp1,Cor1);
set(pl1,{'linewidth'},{1,3}');
%% final training
spr = 0.242;
fintrain = [trainInp valInp];
finT = [T valT];
net = newpnn(fintrain,ind2vec(finT),spr);
% simulate PNN on training data
finY = sim(net,fintrain);
% convert PNN outputs
finY = vec2ind(finY);
% calculate [%] of correct classifications
finCor = 100 * length(find(finT==finY)) / length(finT);
fprintf('\nSpread = %.3f\n',spr)
fprintf('Num of neurons = %d\n',net.layers{1}.size)
fprintf('Correct class = %.3f %%\n',finCor)
% plot targets and network response
figure;
plot(T')
ylim([0 4])
set(gca,'ytick' ,[1 2 3])
hold on
grid on
plot(Y','r')
legend('Targets','Network response')
xlabel('Sample No.')
%% Testing
rand_ind = randperm(50);
testSeto = meas(rand_ind(36:50),:);
testSeto=testSeto';
testVers = meas(50 + rand_ind(36:50),:);
testVers=testVers';
testVirg = meas(100 + rand_ind(36:50),:);
testVirg=testVirg';

% define test set
testInp = [testSeto testVers testVirg];

temp1=repmat(a,1,length(testSeto));
temp2=repmat(b,1,length(testVers));
temp3=repmat(c,1,length(testVirg));
testT = [temp1 temp2 temp3];

testOut = sim(net,testInp);
testOut = vec2ind(testOut);
testCor = 100 * length(find(testT==testOut)) / length(testT);
fprintf('\nSpread = %.3f\n',spr)
fprintf('Num of neurons = %d\n',net.layers{1}.size)
fprintf('Correct class = %.3f %%\n',testCor)
% plot targets and network response
figure;
plot(testT')
ylim([0 4])
set(gca,'ytick' ,[1 2 3])
hold on
grid on
plot(testOut','r')
legend('Targets','Network response')
xlabel('Sample No.')
clc;
clear;

%%  data processing ���ݴ���
load data.mat
dataTrain = trainData(: ,1:8)';
labelTrain = trainData(: ,9)';
dataTest = testData(: ,1:8)';
labelTest = testData(: ,9)';
%�����ݽ��й�һ��
[dataTrain_n, dataTrain_ps] = mapminmax(dataTrain, 0, 1); 
dataTest_n = mapminmax('apply',dataTest, dataTrain_ps);
labelTrain_n = (labelTrain .* 2) - 1; %��һ������-1��1������
labelTest_n = (labelTest .* 2) - 1; %��һ������-1��1������
labelTest_n = labelTest_n';
labelTrain_n = labelTrain_n';
dataTrain_n = dataTrain_n';
dataTest_n = dataTest_n';
%% main program ������
itersValue = 200; 
tic
disp('---����ʼִ��---')
disp('---���ڽ���200�ε���ѵ��---')
[w_0, w, v] = stocGradAscent(dataTrain_n, labelTrain_n, 20, itersValue);
accuary1 = getAccuracy(dataTrain_n, labelTrain_n, w_0, w, v);
display1 = ['ѵ��������׼ȷ��Ϊ��', num2str(accuary1)];
disp(display1)
accuary2 = getAccuracy(dataTest_n, labelTest_n, w_0, w, v);
display2 = ['���Լ�����׼ȷ��Ϊ��', num2str(accuary2)];
disp(display2)
disp('---����ִ�����---')
toc

%% sigmoid function
function y = sigmoid(inx)
y = 1 / (1+exp(-inx));
end

%% FM algorthim
function [w_0, w, v] = stocGradAscent(data, label, k, iters)
dim = size(data);
row = dim(1); %�У���������
col = dim(2); %�У�������Ŀ
alpha = 0.01;
w = zeros(col,1);
w_0 = 0;
v = normrnd(0, 0.2) * ones(col, k); %n��k�е���̬�ֲ���������� 

for it = 1:iters  
    for x = 1:row %x�Ǳ�ʾ��ÿһ�У�matlab������һ��Ҫ��data(1, :)
        inter_1 = data(x, :) * v; %�˴�data(x)����������data����ĵ�x�еĵ�һ��ֵ
        %data(x, :)��1��n�еľ���v���������n��k�� inter_1Ϊ1��k�� inter_2Ҳ��1��k��
        inter_2 = (data(x, :) .* data(x, :)) * (v .* v);
        inter_sum = sum(sum((inter_1 .* inter_1) - inter_2)) ./ 2; 
        output = w_0 + data(x, :) * w + inter_sum;
        loss = sigmoid(label(x) * output(1,1)) - 1;
        w_0 = w_0 - alpha * loss * label(x);
        for i = 1:col
            if data(x,i) ~= 0
                w(i, 1) = w(i, 1) - alpha * loss * label(x) * data(x, i);
                for j = 1:k
                    v(i, j) = v(i, j) - alpha * loss * label(x) *...
                        (data(x, i) * inter_1(1, j) - v(i, j) * data(x, i)...
                        .* data(x, i));
                end
            end
        end
    end 
end
end

%% get accuracy for FM
function accuracy = getAccuracy(data, label, w_0, w, v)
dim = size(data);
row = dim(1);
dataSum = 0;
error1 = 0;
error2 = 0;
for x = 1:row
    dataSum = dataSum + 1;
    inter_1 = data(x, :) * v;
    inter_2 = (data(x, :) .* data(x, :)) * (v .* v);
    inter_sum = sum(sum((inter_1 .* inter_1) - inter_2)) / 2;
    output = w_0 + data(x, :) * w + inter_sum;
    pre = sigmoid(output);

    if(pre < 0.5 && label(x) == 1)
        error1 = error1 + 1;
    elseif (pre >= 0.5 && label(x) == -1)
        error2 = error2 + 1;
    else
        continue
    end
end
disp(error1)
disp(error2)
accuracy =1 - ((error1+error2) / dataSum);
end







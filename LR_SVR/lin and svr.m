clear all;
clc;
data=csvread('output.csv'); 
s=size(data);

 %preprocessed data from "output.csv" you dont't need to prepare for the data again. 
load('ppdata.mat')


thd=floor(0.8*s(1));
% generate for traing set and test set.
pp_train=ppdata(1:thd,:);
pp_test=ppdata(thd+1:end,:);
X_train=pp_train(:,1:5);
X_test=pp_test(:,1:5);

%initialize the mape and acc matrix
mape_linear=zeros(1,5);
acc_linear=zeros(1,5);
mape_svr=zeros(1,5);
acc_svr=zeros(1,5);

% set the interval for training and prediction. 
for deltat=1:5;
    y_train=pp_train(:,5+deltat);
    y_test=pp_test(:,5+deltat);
    
    
    % Linear regression to predict y if 
    X_train1=[ones(length(y_train),1) X_train];
    [b,bint,r,rint,stats]=regress(y_train,X_train1);
    X_test1=[ones(length(y_test),1) X_test];
    %linear regression prediction
    y_predict=X_test1*b;
    [mape_linear(:,delta),acc_linear(:,delta)]=eva(y_predict, y_test);
    %train svm and use the model to predict. there are total 5 models.
    model = svmtrain(y_train,X_train,'-s 3 -t 2 -c 2.2 -g 2.8 -p 0.01 -h 0');
    %SVR prediction
    py = svmpredict(y_test,X_test,model);
    [mape_svr(:,delta),acc_svr(:,delta)]=eva(py, y_test);
    path=strcat('model',num2str(deltat)); %there are 5 models for svr.
    save(path,'model','py','y_test');
end







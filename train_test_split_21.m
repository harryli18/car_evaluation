
% We adopted and modified W. Piraya's code form on Gitub for preprocessing. 

urlwrite('http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', 'car.data');
original_data = importdata('car.data');
[data, label] = readData(original_data);

% One hot encoder is applied to turn our dataset from 6 variables to 21
% variables 
[onehot_data, onehot_label] = readDataOneHot_21(original_data);
data = [onehot_data onehot_label];

[m n] = size(data);

% We use 80% of the data as training set and 20% as test set, we proceed to
% save the workspace to the Train_test_21.mat file to ensure we use
% identical data for both ML models.
P = 0.8;
idx = randperm(m);
training = data(idx(1:round(P*m)),:);
testing = data(idx(round(P*m)+1:end),:);

Xtrain = training(:,1:end-1);
Xtest = testing (:,1:end-1);

Ytrain = training(:,end);
Ytest = testing(:,end);



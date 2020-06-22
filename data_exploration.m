% This code runs in matlab R2018B
% This code requires you to be connected to the internet, otherwise will cause an error.
urlwrite('http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', 'car.data');
% Original data is a 1728 * 1 cell. This is converted to 1728 * 6 table of high level categorical predictors and 
% a 1728 * 1 array of numeric classification labels using the readData function, 
% note. The readData function was written by reference one and is reused here for
% preprocessing purpose only.
% the 6 predictors and their respective subattributes are converted to 21
% binary features using the function readDataOneHot_21. This function was written by 
% reference one and is reused here for preprocessing purpose only.
original_data = importdata('car.data');
[data, label] = readData(original_data);
[onehot_data, onehot_label] = readDataOneHot_21(original_data);
%%
% Below code creates a bar graph of counts for the 21 created features all counts are a proportion of the 1728 initial observations. 
y = sum(onehot_data,1);
figure(1);
c = categorical({'buying high','buying low','buying med','buying vhigh','maint high','maint low','maint med','maint vhigh','doors 2','doors 3','doors 4','doors 5more','persons 2','persons 4','persons more','lug boot big','lug boot med','lug boot small','safety high','safety low','safety med'});
h = bar(c,y);
xlabel('21 Engineered Features');
ylabel('Feature Count');
title('Car Feature Counts [from 1728 total instances]');
xtickangle(90);
%%
% the below code block coverts initial data cell into a table for passing to a classification ensemble.
% it provides a useful visual metric with regard to relative feature
% importance of 6 initial predictors in the car evaluation decision.

data = array2table(data);
ens = fitcensemble(data,label);
imp = predictorImportance(ens);
c = categorical({'buying','maint','doors','persons','lug boot','safety'});
figure(2);
bar(c,imp)
xlabel('6 High level features');
ylabel('Importance metric');
title('Indicative High-level Feature Importance');
xtickangle(90)

%%
% the below code block coverts one hot data with 21 variables into a table for passing to a classification ensemble.
% it provides a useful visual metric with regard to relative feature
% importance of the 21 engineered predictors for car evaluation decision
% making.

onehotdata = array2table(onehot_data);
ens_2 = fitcensemble(onehotdata,label);
imp_2 = predictorImportance(ens_2);
c = categorical({'buying high','buying low','buying med','buying vhigh','maint high','maint low','maint med','maint vhigh','doors 2','doors 3','doors 4','doors 5more','persons 2','persons 4','persons more','lug boot big','lug boot med','lug boot small','safety high','safety low','safety med'}); 
figure(3);
bar(c,imp_2)
xlabel('21 Car features');
ylabel('Feature Importance metric');
title('Indicative Engineered Feature Importance');
xtickangle(90)



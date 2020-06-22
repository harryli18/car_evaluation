% This code runs with matlab R2018 B
% Train_test.mat contains the identical training and test dataset used for both models. 
% The dataset has been processed by one hot encoder in order to convert cateegorical 
% variables to numeric for use in our machine learning models.

load("Train_test_21.mat")


%% 
% The below code gets the data ready (predictor and class data) to pass
% into both our standard bagged decision tree model and our tuned random forest.
% bayesian optimisation in order tolater tune model hyperparameters;
%
Xt = array2table(Xtest);
Yt = table(Ytest);
Xt = [Xt Yt];
X = array2table(Xtrain);
X.Properties.VariableNames = {'buying_high','buying_low','buying_med','buying_vhigh','maint_high','maint_low','maint_med','maint_vhigh','doors_2','doors_3','doors_4','doors_5more','persons_2','persons_4','persons_more','lug_boot_big','lug_boot_med','lug_boot_small','safety_high','safety_low','safety_med'};
Y2 = table(Ytrain);
X = [X Y2];
rng('default'); % For reproducibility

% Note: Data preprocssing for modelling purposes finishes here.

%%
% Train a standard tree bagger model in this classification problem which implements 
% bootstrap aggregation, effectively creating many trees and sampling the data with substitution.
% note upon hyperparamter tuning, if less than all predictors are used in
% each tree then we effectively toggle from bagged decision tree to a random
% forest.
Mdl = TreeBagger(300,X,'Ytrain','Method','classification','OOBPrediction','on');
err = oobError(Mdl);
MdlMeanOOBErr = mean(oobError(Mdl));

% Make a prediction for the test set using the standard treebagger model
% Note training has taken place in the above code block.
[Y_Mdl, classifScore] = Mdl.predict(Xtest);
Y_Mdl = str2double(Y_Mdl);

[label,score] = oobPredict(Mdl);   
% Compute the confusion matrix
C_Mdl = confusionmat(Ytest,Y_Mdl)
C_Mdlp = bsxfun(@rdivide,C_Mdl,sum(C_Mdl,2)) * 100

%%
% Here we prepare and illustrate a confusion matrix showing the class
% labels predicted by our model versus the true class labels from the test
% set

Y_Mdlc = Y_Mdl.';
Ytestc = Ytest.';
pod = sum(Ytestc(:) == 1); % only used as a check for class counts.
figure(1);
cm = confusionmat(Ytestc,Y_Mdlc);
clabels = {'0 Unnaceptable car','1 Acceptable car','2 Good car','3 Very good car'};
cmc = confusionchart(cm,clabels);
title('CM for Non-tuned bootstrap aggregation model');

%%
% Here we graphically plot one of the trees for information and interpretation (not used in
% poster as too unwieldy to illustrate)
Yfit = predict(Mdl,X);
Tree10 = Mdl.Trees{10};
view(Tree10,'Mode','graph');

%%

% In subsequent code (below next 2 code blocks) we complete hyperparameter tuning on 
% 1. number of observations per leaf & 
% 2. Number of predictors to sample from.

% Because the impact of tuning can be quite difficult to interpret, in the
% next 2 code blocks we graphically plot out of bag error on training data for various
% fixed values of the 2 hyperparameters and plot with respect to all grown trees.
% We should expect the output of these plots to agree with the tabular results of hyper-parameter tuning.

leaf = [1 3 7 10];
nTrees = 300;
rng(9876,'twister');
savedRng = rng; % save the current RNG settings

color = 'bgrc';
for i = 1:length(leaf)
   % Reinitialize the random number generator, so that the
   % random samples are the same for each leaf size
   rng(savedRng);
   % Create a bagged decision tree for each leaf size and plot out-of-bag
   % error 'oobError'
    Mdl = TreeBagger(nTrees,X,'Ytrain','Method','classification',...
    'MinLeafSize',leaf(i),'OOBPrediction','on');
   figure(2);
   plot(Mdl.oobError,color(i));
   hold on;
end
xlabel('Number of grown trees');
ylabel('Out-of-bag classification error');
legend({'1', '3', '7','10'},'Location','NorthEast');
title('Classification Error for Different Leaf Sizes');
hold off;

%%
% Plot of oob error on training data versus trees grown, for parameter
% 'number of predictors to sample'
Nptsc = [6 12 19 21];
nTrees = 300;
rng(9876,'twister');
savedRng = rng; % save the current RNG settings

color = 'bgrc';
for ii = 1:length(Nptsc)
   % Reinitialize the random number generator, so that the
   % random samples are the same for each leaf size
   rng(savedRng);
   % Create a bagged decision tree for each leaf size and plot out-of-bag
   % error 'oobError'
    Mdl = TreeBagger(nTrees,X,'Ytrain','Method','classification',...
    'NumPredictorstoSample',Nptsc(ii),'OOBPrediction','on');
   figure(3);
   plot(Mdl.oobError,color(ii));
   hold on;
end
xlabel('Number of grown trees');
ylabel('Out-of-bag classification error');
legend({'6', '12', '19','21'},'Location','NorthEast');
title('Classification Error for varying predictor counts');
hold off;
%%
% Below we specify paramters for tuning and proceed to do so using bayesian
% optimisation process. We Cannot tune based on quantile error in our case so must specify mean error for minimisation, since this is
% a classification problem. Hypeparamters need to be tuned before learning starts.
maxMinLS = 20;
minLS = optimizableVariable('minLS',[1,maxMinLS],'Type','integer');
numPTS = optimizableVariable('numPTS',[1,size(X,2)-1],'Type','integer');
hyperparametersRF = [minLS; numPTS];
%%
% Minimize Objective Using Bayesian Optimization and OOBErrRF function 
results = bayesopt(@(params)oobErrRF(params,X),hyperparametersRF,...
    'AcquisitionFunctionName','expected-improvement-plus','Verbose',1);
%%
% Display the observed minimum of the objective function and the optimized hyperparameter values.

bestOOBErr = results.MinObjective;
bestHyperparameters = results.XAtMinObjective;

%%
% Below we train our model using the newly optimized hyperparamters.
Mdlh = TreeBagger(300,X,'Ytrain','Method','classification',...
    'MinLeafSize',bestHyperparameters.minLS,...
    'NumPredictorstoSample',bestHyperparameters.numPTS,...
    'OOBPrediction','on');

errh = oobError(Mdlh);
MdlMeanOOBErrh = mean(oobError(Mdlh));

%%

% Make a prediction for the test set using our now tuned model.
[Y_Mdlh, classifScore] = Mdlh.predict(Xtest);
Y_Mdlh = str2double(Y_Mdlh);
[labelh,scoreh] = oobPredict(Mdlh);

% Compute the confusion matrix
C_Mdlh = confusionmat(Ytest,Y_Mdlh)

% Examine confusion matrix for each of the the 4 classes as a percentage of
% the true class
C_Mdlhp = bsxfun(@rdivide,C_Mdlh,sum(C_Mdlh,2)) * 100

%%

% Again prepare and illustrate a confusion matrix showing the class
% labels predicted by tuned model versus the true class labels from the
% test set. This enables us to compare confusion matrices from the standard
% and tuned models on our poster.

Y_Mdlhc = Y_Mdlh.';
Ytesthc = Ytest.';
figure(6);
cmh = confusionmat(Ytesthc,Y_Mdlhc);
chlabels = {'0 Unnaceptable car','1 Acceptable car','2 Good car','3 Very good car'};
cmhc = confusionchart(cmh,chlabels);
title('CM for Tuned Random forest model');

%%

% Make a plot showing relative predictor importance. Treebagger is a matlab
% bootstrap aggregation implemenation which offers up useful functionality
% in plotting feature importance, particularly useful for car evaluation
% dataset

n = 300;
leafs = 1;
rng(savedRng);

b = TreeBagger(n,Xt,'Ytest','Method','classification','OOBVarImp','on',...
                          'CategoricalPredictors',21,...
                          'MinLeaf',leafs);
figure(7);
c = categorical({'buying high','buying low','buying med','buying vhigh','maint high','maint low','maint med','maint vhigh','doors 2','doors 3','doors 4','doors 5more','persons 2','persons 4','persons more','lug boot big','lug boot med','lug boot small','safety high','safety low','safety med'});
bar(c,b.OOBPermutedPredictorDeltaError);
xtickangle(90)
xlabel('Car Features');
ylabel('Out-of-bag feature importance');
title('Feature importance results');

oobErrorFullXt = b.oobError;


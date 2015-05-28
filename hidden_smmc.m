addpath('/Users/gangchen/Downloads/RBM/tSNE/Parametric-t-SNE')


yy=load('sonar_m.txt');
x=yy(:,1:60)'; %  sonar [60 x 208]
correctLabel=yy(:,61);
clear yy

%
[dim, nPoints] = size(x);
for i = 1 : nPoints
    X(:,i) = x(:,i) - (mean(x'))';  
end
continous =1;

training = 'CD1';
LEARN = false;
modelname = 'initmodel0.mat';

train_X = X';
% layers = [500 500 2000 2];
layers = 64;
L = numel(layers);
network = init_layer_dbn(train_X, layers, L, continous, training, LEARN, modelname);



nRepeatsPerConstraintSet =2; %20;
nConstraint_list = [100];%[20 40 60 80 100];

nClusters = length(unique(correctLabel));


for cidx = 1: length(nConstraint_list) 

    nPairs = nConstraint_list(cidx);
    fprintf('-------------- %d pairs of constraints -------------\n\n',nPairs);

    for nRepeat = 1 : nRepeatsPerConstraintSet
        consGenOpt.pseudo_random = 0;      
        consGenOpt.balance = 0; 
        T = translateLabelsToConstraints(nPairs,correctLabel,consGenOpt);
       
        %
        %$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        % constrainedMMC_subGra_tkde
        %$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        lambda = 1e-2;
        cOne = 1;

        option.initMethod = 'pda'; 
        option.trueLabel = correctLabel;
        option.preIteration = 2; %t=0:preIteration
        option.innerTol = 0.01;
        option.perQuit = 0.01;

        % embeddings
        % mappedX = run_data_through_network(network, train_X);
        [clusterLabel, W,  CCCP_objFunVal, elapsedTime] = constrainedMMC_hidden(double(train_X),network, nClusters, lambda, cOne, T, option);
        % [clusterLabel, W,  CCCP_objFunVal, elapsedTime] = constrainedMMC_hidden2(double(train_X),network, nClusters, lambda, cOne, T, option);
        fprintf('======== constrainedMMC_tkde with %d pairs of constraints( repeat %d finished) ========\n\n',nPairs, nRepeat);       
        
    end

end
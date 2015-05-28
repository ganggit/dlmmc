
yy=load('sonar_m.txt');
X=yy(:,1:60)'; %  sonar [60 x 208]
labels=yy(:,61);
clear yy

%
[dim, nPoints] = size(X);

continous =1;

training = 'CD1';
LEARN = false;% true; % 
modelname = 'initmodel3.mat';

train_X = X;
uLabels = unique(labels);
nRepeatsPerConstraintSet =2; %20;
nConstraint_list = [100];%[20 40 60 80 100];

nClusters = length(uLabels);


for cidx = 1: length(nConstraint_list) 

    nPairs = nConstraint_list(cidx);
    consGenOpt.pseudo_random = 0;      
    consGenOpt.balance = 0; 
    % generate pairwise matching matrix T
    T = translateLabelsToConstraints(nPairs,labels,consGenOpt);


    [fullI_C_vec, fullJ_C_vec] = find( T == -1 );
    I_C_vec = [];J_C_vec=[];
    for i = 1:length(fullI_C_vec)
        if fullI_C_vec(i) > fullJ_C_vec(i) % remove redundance caused by symmetry
            I_C_vec = [I_C_vec fullI_C_vec(i)];
            J_C_vec = [J_C_vec fullJ_C_vec(i)];
        end
    end  

    %**************************************************************************
    % index for (i,j)\in M  (Note: i > j)
    %**************************************************************************
    [fullI_M_vec, fullJ_M_vec] = find( T == 1 );
    I_M_vec = [];J_M_vec=[];
    for i = 1:length(fullI_M_vec)
        if fullI_M_vec(i) > fullJ_M_vec(i) % remove redundance caused by symmetry
            I_M_vec = [I_M_vec fullI_M_vec(i)];
            J_M_vec = [J_M_vec fullJ_M_vec(i)];
        end
    end


    %**************************************************************************
    % index for i \nin {M U C} 
    %**************************************************************************
    I_MC_vec = [I_C_vec J_C_vec I_M_vec J_M_vec];
    I_NonMC_vec = 1:nPoints;
    I_NonMC_vec(I_MC_vec) = []; % the set U=I_NonMC_vec

    
    % for the structure transformation
    idx = 1;
    pairs = [];
    for i=1:length(I_M_vec)
        
        pairs(idx).pairId = idx-1;
        pairs(idx).img1 = struct('id', I_M_vec(i), 'classId', labels(I_M_vec(i)));
        pairs(idx).img1 = struct('id', J_M_vec(i), 'classId', labels(J_M_vec(i)));
        pairs(idx).match =1;
        pairs(idx).training =1;
        pairs(idx).fold =1;
        idx= idx+1;
    end
    
    for i=1:length(I_C_vec)
        
        pairs(idx).pairId = idx-1;
        pairs(idx).img1 = struct('id', I_C_vec(i), 'classId', labels(I_C_vec(i)));
        pairs(idx).img1 = struct('id', J_C_vec(i), 'classId', labels(J_C_vec(i)));
        pairs(idx).match =0;
        pairs(idx).training =1;
        pairs(idx).fold =1;
        idx = idx+1;
    end

     
    
    idxa = [I_M_vec, I_C_vec]';
    idxb = [J_M_vec, J_C_vec]';
    
% -----------------------------------------------------------
% method Kmeans
% -----------------------------------------------------------




% -----------------------------------------------------------
% method CPMMC 
% -----------------------------------------------------------
addpath('/Users/gangchen/Downloads/clustering/CPMMC');




%% Xing metric learning 
addpath('/Users/gangchen/Downloads/learning/metriclearning/code_Metric_online');
addpath('/Users/gangchen/Downloads/RBM/Autoencoder_Code');
% need to define S and D
S = T==1;
D = T==-1;
maxiter =50;
A = opt(train_X', S, D, maxiter);


% show the ROC
matches = logical([pairs.match]);
dist = cdistM(A,X,idxa,idxb); 
[tpr, fpr] = icg_roc(matches,-dist);
[ignore, eerIdx] = min(abs(tpr - (1-fpr)));
%eer
eer =tpr(eerIdx);
        
h = icg_plotroc(matches,-dist);
hold on; plot(fpr,tpr,'Color','r','LineWidth',2,'LineStyle','-'); hold off;

% show the clustering performance
Kdist = computeMdist(train_X, A);
Z = kernelkmeans(Kdist, nClusters, maxiter); 
prelabel = vec2index(Z);
% generage pairs for this method
Accuracy = evaluate_Acc(prelabel',nClusters,  option);

% -------------------------------------------------------------------
% itml & kissme
% -------------------------------------------------------------------

%% method itml
addpath('/Users/gangchen/Downloads/clustering/KISSME');
%% method KISSME
addpath('/Users/gangchen/Downloads/learning/metriclearning/itml2');
params.pca.numDims = 50; %we project onto the first 50 PCA dim. 
pair_metric_learn_algs = {LearnAlgoITML(),LearnAlgoKISSME()};
ds = CrossValidatePairs(struct(),pair_metric_learn_algs, pairs, train_X, idxa, idxb);   
% ds = CrossValidatePairs(ds,metric_learn_algs,pairs, ux(1:params.pca.numDims,:), idxa, idxb, @ToyCarPairsToLabels); 

% EVALUATION one the pairwise performance 
% we evaluate only on the test set (fold 2), train set (fold 1).
[ignore, rocPlot] = evalData(pairs(logical([pairs.training])), ds(1), params);
hold on; plot(1-0.859,0.859,'+','Color',[0.5 0.5 1],'LineWidth',2);
legendEntries = get(rocPlot.hL,'String');
legendEntries{end+1} = 'Nowak (0.859)';
legend(gca(rocPlot.h),legendEntries,'Location', 'SouthEast');
title('ROC Curves ToyCars');

if isfield(params,'saveDir')
    exportAndCropFigure(rocPlot.h,'all_toycars',params.saveDir);
    save(fullfile(params.saveDir,'all_data.mat'),'ds');
end


% evaluate one the clustering performance 
matches = logical([pairs.match]);
%-- EVAL FOLDS --%
un = unique([pairs.fold]);
Accuracy = zeros(length(un), length(names));
for c=1:length(un)
    testMask = [pairs.fold] == un(c);  

    % eval fold
    names = fieldnames(ds(c));
    for nameCounter=1:length(names)
        % get matrix
        M = ds(c).(names{nameCounter}).M;
        Kdist = computeMdist(train_X, A);
        Z = kernelkmeans(Kdist, nClusters, maxiter); 
        prelabel = vec2index(Z);
        % generage pairs for this method
        Accuracy(c, nameCounter) = evaluate_Acc(prelabel',nClusters,  option);
    end
end



% itde from hong kong university 
% -----------------------------------------------------------------
% constrainedMMC_subGra_tkde
% -----------------------------------------------------------------
lambda = 1e-2;
cOne = 1;

option.initMethod = 'pda'; 
option.trueLabel = labels;
option.preIteration = 2; %t=0:preIteration
option.innerTol = 0.01;
option.perQuit = 0.01;

% embeddings
% mappedX = run_data_through_network(network, train_X);
[clusterLabel, W,  CCCP_objFunVal, elapsedTime] = constrainedMMC_tkde(double(train_X), nClusters, lambda, cOne, T, option);

Accuracy = evaluate_Acc(clusterLabel,nClusters, option);

[M_val, C_val] = evaluate_pairwise(X, W, nClusters, I_C_vec, J_C_vec, I_M_vec, J_M_vec);
dist = [M_val, C_val];
[tpr, fpr] = icg_roc(matches, dist);
[ignore, eerIdx] = min(abs(tpr - (1-fpr)));
%eer
eer =tpr(eerIdx);
        
h = icg_plotroc(matches,dist);
hold on; plot(fpr,tpr,'Color','r','LineWidth',2,'LineStyle','-'); hold off;

% ----------------------------------------------------------------
% our method
% ----------------------------------------------------------------

% layers = [500 500 2000 2];
layers = 64;
layers = [layers, length(unique(correctLabel))];
L = numel(layers);
network = init_layer_dbn(train_X, layers, L, continous, training, LEARN, modelname);

[clusterLabel, W,  CCCP_objFunVal, elapsedTime] = constrainedMMC_dbn_2(double(train_X'),network, nClusters, lambda, cOne, T, option);

Accuracy = evaluate_Acc(clusterLabel,nClusters, option);
% projection
temp = projection(network, train_X', L-1);
[M_val, C_val] = evaluate_pairwise(temp', W, nClusters, I_C_vec, J_C_vec, I_M_vec, J_M_vec);
dist = [M_val, C_val];
[tpr, fpr] = icg_roc(matches, dist);
[ignore, eerIdx] = min(abs(tpr - (1-fpr)));
%eer
eer =tpr(eerIdx);
        
h = icg_plotroc(matches,dist);
hold on; plot(fpr,tpr,'Color','r','LineWidth',2,'LineStyle','-'); hold off;


end
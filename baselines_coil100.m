function baselines_coil20(nConstraint)

close all;

addpath('/gpfs/scratch/gangchen/GangChen_CVPR/RBMLIB/DPMM');
addpath('/panfs/panfs.ccr.buffalo.edu/scratch/GangChen_CVPR/RBMLIB/spectral-0.2');
%% method itml
addpath(genpath('/gpfs/scratch/gangchen/GangChen_CVPR/dmmc/clustering/KISSME'));
%% method KISSME
addpath('/gpfs/scratch/gangchen/GangChen_CVPR/dmmc/clustering/itml2');

dataid = 'coil_100';
% fpath = '/panfs/panfs.ccr.buffalo.edu/scratch/GangChen_CVPR/dataset/';
fpath = '/gpfs/scratch/gangchen/GangChen_CVPR/dmmc/tkde12_cmmc_code/';
fname = 'coil_100.mat';
load(fullfile(fpath, fname));
X=double(data')./255; %  sonar [60 x 208]
%labels=la;


%
[dim, nPoints] = size(X);

continous =0;

training = 'CD1';
LEARN = true; %false;%
modelname = [dataid '.mat'];

train_X = X;
uLabels = unique(labels);
nRepeatsPerConstraintSet =2; %20;


% nConstraint_list = [100];%[20 40 60 80 100];
if nargin <1
    nConstraint_list = [100];%[20 40 60 80 100];
else
    nConstraint_list=str2num(nConstraint);
end

nClusters = length(uLabels);

% initialize parameters
option.initMethod = 'pda'; 
option.trueLabel = labels;
option.preIteration = 2; %t=0:preIteration
option.innerTol = 0.01;
option.perQuit = 0.01;
drawcolors ={'y', 'm', 'c', 'r', 'g', 'b'}; 
Accuracy = cell(length(drawcolors), 1);
rocval = cell(length(drawcolors), 1);
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
%    tests with random samplings 
% -----------------------------------------------------------

[I_C_test, J_C_test, I_M_test, J_M_test, testpairs]=  generateconstraints(nPairs,labels, consGenOpt);
idxa_test = [I_M_test, I_C_test]';
idxb_test = [J_M_test, J_C_test]';  
    
% -----------------------------------------------------------
% method Kmeans
% -----------------------------------------------------------




% -----------------------------------------------------------
% method CPMMC 
% -----------------------------------------------------------
addpath('/Users/gangchen/Downloads/clustering/CPMMC');


maxiter =50;
% -----------------------------------------------------------
%% Xing metric learning 
% -----------------------------------------------------------
iflag = true;%false; %


% figure related here
icolor = 0; % index for color here

h = figure(1);
hold on; 
title(['ROC:' dataid]);

ylabel('True Positive Rate (TPR)');
xlabel('False Positive Rate (FPR)');

grid on;
xlim([0 1]);
ylim([0 1]);


if iflag 

addpath('/gpfs/scratch/gangchen/GangChen_CVPR/dmmc/clustering/code_Metric_online');
addpath('/gpfs/scratch/gangchen/GangChen_CVPR/RBMLIB/Autoencoder_Code');
% need to define S and D
S = T==1;
D = T==-1;

% training the model: Mahanobios matrix
% A = opt(double(train_X'), S, D, maxiter);
A = opt(double(train_X'), S, D, 4);
matches = logical([testpairs.match]);
dist = cdistM(A,X,idxa_test,idxb_test); 
[tpr, fpr] = icg_roc(matches,-dist);
[ignore, eerIdx] = min(abs(tpr - (1-fpr)));
%eer
eer =tpr(eerIdx);
% show the ROC       
% h = icg_plotroc(matches,-dist);
icolor = icolor + 1; 
% hold on; 
plot(fpr,tpr,'Color',drawcolors{icolor},'LineWidth',2,'LineStyle','-'); 
% hold off;
rocval{icolor}.tpr = tpr;
rocval{icolor}.fpr = fpr;
% show the clustering performance
Kdist = computeMdist(train_X, A);
Z = kernelkmeans(Kdist, nClusters, maxiter); 
prelabel = vec2index(Z);
% generage pairs for this method
% prelabel = spclustering(Kdist,nClusters,100,0.0);
Accuracy{icolor}.rand = RandIndex(option.trueLabel, prelabel);
Accuracy{icolor}.acc = evaluate_Acc(prelabel',nClusters,  option);

strnum = num2str(nConstraint_list);
save([dataid '_' strnum '_kernel.mat'], 'Accuracy', 'rocval');
end
% -------------------------------------------------------------------
% itml & kissme
% -------------------------------------------------------------------
if iflag
%% method itml
addpath(genpath('/gpfs/scratch/gangchen/GangChen_CVPR/dmmc/clustering/KISSME'));
%% method KISSME
addpath(genpath('/gpfs/scratch/gangchen/GangChen_CVPR/dmmc/clustering/itml2'));
params.pca.numDims = 50; %we project onto the first 50 PCA dim. 
pair_metric_learn_algs = {LearnAlgoITML(),LearnAlgoKISSME()};
ds = CrossValidatePairs(struct(),pair_metric_learn_algs, pairs, train_X, idxa, idxb);   
% ds = CrossValidatePairs(ds,metric_learn_algs,pairs, ux(1:params.pca.numDims,:), idxa, idxb, @ToyCarPairsToLabels); 

% % EVALUATION one the pairwise performance 
% % we evaluate only on the test set (fold 2), train set (fold 1).
% [ignore, rocPlot] = evalData(pairs(logical([pairs.training])), ds(1), params);
% hold on; plot(1-0.859,0.859,'+','Color',[0.5 0.5 1],'LineWidth',2);
% legendEntries = get(rocPlot.hL,'String');
% legendEntries{end+1} = 'Nowak (0.859)';
% legend(gca(rocPlot.h),legendEntries,'Location', 'SouthEast');
% title('ROC Curves ToyCars');
% 
% if isfield(params,'saveDir')
%     exportAndCropFigure(rocPlot.h,'all_toycars',params.saveDir);
%     save(fullfile(params.saveDir,'all_data.mat'),'ds');
% end


% evaluate one the clustering performance 
matches = logical([testpairs.match]);
%-- EVAL FOLDS --%
un = unique([testpairs.fold]);
% Accuracy = zeros(length(un), length(pair_metric_learn_algs));
for c=1:length(un)
    testMask = [testpairs.fold] == un(c);  

    % eval fold
    names = fieldnames(ds(c));
    % Acc = zeros(length(un), length(names));
    for nameCounter=1:length(names)
        % get matrix
        M = ds(c).(names{nameCounter}).M;
        
        dist = cdistM(M,X,idxa_test,idxb_test); 
        [tpr, fpr] = icg_roc(matches,-dist);
        [ignore, eerIdx] = min(abs(tpr - (1-fpr)));
        %eer
        eer =tpr(eerIdx);
        icolor = icolor + 1;
        % h = icg_plotroc(matches,-dist);
        % hold on; 
        plot(fpr,tpr,'Color',drawcolors{icolor},'LineWidth',2,'LineStyle','-'); 
        % hold off;
        
        Kdist = computeMdist(train_X, M);
        Z = kernelkmeans(Kdist, nClusters, maxiter); 
        prelabel = vec2index(Z);
        % prelabel = spclustering(Kdist,nClusters,100,0.0);
        Accuracy{icolor}.rand = RandIndex(option.trueLabel, prelabel);
        % generage pairs for this method
        Accuracy{icolor}.acc = evaluate_Acc(prelabel',nClusters,  option);
        rocval{icolor}.tpr = tpr;
        rocval{icolor}.fpr = fpr;
    end
    
end
end
strnum = num2str(nConstraint_list);
save([dataid '_' strnum '_kernel.mat'], 'Accuracy', 'rocval');

% itde from hong kong university 
% -----------------------------------------------------------------
% constrainedMMC_subGra_tkde
% -----------------------------------------------------------------
lambda = 1e-2;
matches = logical([testpairs.match]);
if iflag
    cOne = 1;
else
    cOne = 0;
end

% embeddings
% mappedX = run_data_through_network(network, train_X);
[clusterLabel, W1,  CCCP_objFunVal, elapsedTime] = constrainedMMC_tkde(double(train_X), nClusters, lambda, cOne, T, option);

Acc = evaluate_Acc(clusterLabel,nClusters, option);
icolor = icolor + 1;
Accuracy{icolor}.acc = Acc;
Accuracy{icolor}.rand = RandIndex(option.trueLabel, clusterLabel);
% % [M_val, C_val] = evaluate_pairwise(X, W, nClusters, I_C_test, J_C_test, I_M_test, J_M_test);
% [C_val, M_val, C_violation, M_violation] = evaluate_pairwise3(X, W, nClusters, I_C_vec, J_C_vec, I_M_vec, J_M_vec);
% dist = [M_val, C_val, M_violation, C_violation];
% gt = [ones(1, length(matches)), zeros(1, length(matches))];
% [tpr, fpr] = icg_roc(gt, dist);

A1 = W1*W1';
dist = cdistM(A1,train_X,idxa_test,idxb_test); 
[tpr, fpr] = icg_roc(matches,-dist);
rocval{icolor}.tpr = tpr;
rocval{icolor}.fpr = fpr;
[ignore, eerIdx] = min(abs(tpr - (1-fpr)));
%eer
eer =tpr(eerIdx);
        
% h = icg_plotroc(matches,dist);
% hold on; 
plot(fpr,tpr,'Color',drawcolors{icolor},'LineWidth',2,'LineStyle','-'); 
% hold off;
save([dataid '_' strnum '_kernel.mat'], 'Accuracy', 'rocval');
% ----------------------------------------------------------------
% our method
% ----------------------------------------------------------------

% layers = [500 500 2000 2];
layers =100;
layers = [layers, length(unique(labels))];
L = numel(layers);
network = init_layer_dbn(train_X', layers, L, continous, training, LEARN, modelname);

[clusterLabel, W,  CCCP_objFunVal, elapsedTime] = constrainedMMC_dbn_2(double(train_X'),network, nClusters, lambda, cOne, T, option);

Acc = evaluate_Acc(clusterLabel,nClusters, option);
icolor = icolor + 1;
Accuracy{icolor}.acc = Acc;
Accuracy{icolor}.rand = RandIndex(option.trueLabel, clusterLabel);
% projection
temp = projection(network, train_X', L-1);
% [M_val, C_val] = evaluate_pairwise(temp', W, nClusters, I_C_test, J_C_test, I_M_test, J_M_test);
% dist = [M_val, -C_val];
% [tpr, fpr] = icg_roc(matches, dist);

% [C_val, M_val, C_violation, M_violation] = evaluate_pairwise3(temp', W, nClusters, I_C_vec, J_C_vec, I_M_vec, J_M_vec);
% dist = [M_val, C_val, M_violation, C_violation];
% gt = [ones(1, length(matches)), zeros(1, length(matches))];
% % gt = [ones(1, length(M_val)), zeros(1, length(M_violation))];
% % dist = [C_val, C_violation];
% dist = [M_val./max(M_val), C_val./max(C_val), M_violation./max(M_val), C_violation./max(C_val)];
% gt = [ones(1, length(matches)), zeros(1, length(matches))];
% [tpr, fpr] = icg_roc(gt, dist);

A = W*W';
dist = cdistM(A,temp',idxa_test,idxb_test); 
[tpr, fpr] = icg_roc(matches,-dist);
rocval{icolor}.tpr = tpr;
rocval{icolor}.fpr = fpr;
[ignore, eerIdx] = min(abs(tpr - (1-fpr)));
%eer
eer =tpr(eerIdx);
        
% h = icg_plotroc(matches,dist);
% hold on; 
plot(fpr,tpr,'Color',drawcolors{icolor},'LineWidth',2,'LineStyle','-'); 


if icolor == 5
legend( 'Xing','ITML','KISSME', 'CMMC', 'Our method');
elseif icolor ==6

    legend( 'k-means', 'Xing','ITML','KISSME', 'CMMC', 'Our method');
end
% add the legend here

hold off;
strnum = num2str(nConstraint_list);
if iflag
save([dataid '_' strnum '_kernel.mat'], 'Accuracy', 'rocval');
saveas(h,[dataid strnum '_roc.pdf'],'pdf') ;
else
    save([dataid '_' strnum '_notran2.mat'], 'Accuracy', 'rocval');
end
end

function dtmmc_demo_coil100(nConstraint)

close all;


dataid = 'coil_100_DBN';
% fpath = '/panfs/panfs.ccr.buffalo.edu/scratch/GangChen_CVPR/dataset/';
fpath = './';
fname = 'coil_100.mat';
load(fullfile(fpath, fname));
X=double(data')./255; %  sonar [60 x 208]
%labels=la;


%
[dim, nPoints] = size(X);

continous =0;

training = 'CD1';
LEARN =  true; %false;%
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
option.perQuit = 0.03; %0.01;
drawcolors ={'y', 'm', 'c', 'r', 'g', 'b'}; 
Accuracy = cell(length(drawcolors), 1);
rocval = cell(length(drawcolors), 1);
for cidx = 1: length(nConstraint_list) 

    nPairs = nConstraint_list(cidx);
    consGenOpt.pseudo_random = 0;      
    consGenOpt.balance = 1;%0; 
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
    
    
end    
% -----------------------------------------------------------
%    tests with random samplings 
% -----------------------------------------------------------

[I_C_test, J_C_test, I_M_test, J_M_test, testpairs]=  generateconstraints(nPairs,labels, consGenOpt);
idxa_test = [I_M_test, I_C_test]';
idxb_test = [J_M_test, J_C_test]';  
    
% -----------------------------------------------------------
% method Kmeans
% -----------------------------------------------------------

iflag = true;%false; %
% -----------------------------------------------------------


lambda = 1e-2;
matches = logical([testpairs.match]);
if iflag
    cOne = 1;
else
    cOne = 0;
end


% figure related here
icolor = 0; % index for color here


% ----------------------------------------------------------------
% our method
% ----------------------------------------------------------------

% layers = [500 500 2000 2];
hier{1} =200;
hier{2} = [400,200];
hier{3} = [400, 200, 100];
hier{4} = [400,300, 200,100];
hier{5} = [400,300, 200,200,100];
for hidx = 1: length(hier)
layers = hier{hidx};
layers = [layers, length(unique(labels))];
L = numel(layers);
train_X = X;
network = init_layer_dbn(train_X', layers, L, continous, training, LEARN, modelname);

[clusterLabel, W,  CCCP_objFunVal, elapsedTime] = dtmmc_constrains(double(train_X'), network, nClusters, lambda, cOne, T, option);

Acc = evaluate_Acc(clusterLabel,nClusters, option);
icolor = icolor + 1;
Accuracy{icolor}.acc = Acc;
Accuracy{icolor}.rand = RandIndex(option.trueLabel, clusterLabel);
% projection
temp = projection(network, train_X', L-1);


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
% plot(fpr,tpr,'Color',drawcolors{icolor},'LineWidth',2,'LineStyle','-'); 

strnum = num2str(nConstraint_list);
save([dataid '_' strnum '_dtmmc.mat'], 'Accuracy', 'rocval');


end

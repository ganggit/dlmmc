clear all, close all,

fpath = '/Users/gangchen/Downloads/RBM/Autoencoder_Code/';

fname = 'digit';%'test';


numcls = 10;

numsamples = 500; % only get the first 200 hundreds digits

digits = cell(2);
numdata = 0;
for i=1:numcls
    
    temp = [fname num2str(i-1) '.mat'];
    temp2 = fullfile(fpath, temp);
    load(temp2);
    
    D = D(1:numsamples,:);
    
    digits{i} = D/255;
    numdata = numdata+size(D,1);
    numdims = size(D,2);
end

X =zeros(numdata, numdims);
y=zeros(numdata,1);
idx = 1;
for i = 1: length(digits)
    [num, numdims] = size(digits{i});
    X(idx:idx+num-1,: ) =  digits{i};
    y(idx:idx+num-1) = i*ones(num, 1);
    idx = idx + num;
end
nRepeatsPerConstraintSet =5; %20;
nConstraint_list = [4000];%[20 40 60 80 100];



% %
% [dim, nPoints] = size(x);
% for i = 1 : nPoints
%     X(:,i) = x(:,i) - (mean(x'))';  
% end
X = X - repmat(mean(X,1), size(X,1),1);

X = X';
correctLabel = y;

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

        [clusterLabel, W,  CCCP_objFunVal, elapsedTime] = constrainedMMC_tkde(X, nClusters, lambda, cOne, T, option);
        fprintf('======== constrainedMMC_tkde with %d pairs of constraints( repeat %d finished) ========\n\n',nPairs, nRepeat);       
        
    end

end


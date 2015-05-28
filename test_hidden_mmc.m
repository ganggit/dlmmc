function test_hidden_mmc


fpath = '/Users/gangchen/Downloads/RBM/Autoencoder_Code/';

fname = 'digit';%'test';
numcls = 10;
numsamples = 500; % only get the first 200 hundreds digits

digits = cell(numcls, 1);
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
correctLabel =y;
continous =0;

training = 'CD1';
LEARN = false;


% layers = [500 500 2000 2];
layers = 64;
L = numel(layers);
network = init_layer_dbn(X, layers, L, continous, training, LEARN);


load(fullfile(fpath, 'mnistvhclassify.mat'));
load(fullfile(fpath,  'mnisthpclassify.mat'));
load(fullfile(fpath, 'mnisthp2classify.mat'));

%%%% PREINITIALIZE WEIGHTS OF THE DISCRIMINATIVE MODEL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% w1=[vishid; hidrecbiases];
% w2=[hidpen; penrecbiases];
% w3=[hidpen2; penrecbiases2];
% w_class = 0.1*randn(size(w3,2)+1,10);
%  
layers = [500 500 2000];

i =1;
network{i}.W = vishid;
network{i}.bias_upW = hidrecbiases;
i =2;
network{i}.W = hidpen;
network{i}.bias_upW = penrecbiases;
i =3;
network{i}.W = hidpen2;
network{i}.bias_upW = penrecbiases2;

nRepeatsPerConstraintSet =2; %20;
nConstraint_list = [4000];%[20 40 60 80 100];

nClusters = length(unique(correctLabel));

train_X =X - repmat( mean(X,1), size(X,1),1);
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
        % [clusterLabel, W,  CCCP_objFunVal, elapsedTime] = constrainedMMC_tkde(train_X', nClusters, lambda, cOne, T, option);
        [clusterLabel, W,  CCCP_objFunVal, elapsedTime] = constrainedMMC_hidden2(double(train_X),network, nClusters, lambda, cOne, T, option);
        fprintf('======== constrainedMMC_tkde with %d pairs of constraints( repeat %d finished) ========\n\n',nPairs, nRepeat);       
        
    end

end
% this function is initially written by Hong Zhen, hzeng@seu.edu.cn


function [clusterLabel, W, CCCP_objFunVal, elapsedTime]=constrainedMMC_tkde(X, nClusters, lambda, cOne, T, option)

if ~isfield(option,'initMethod')
    initMethod = 'pda';
else
    initMethod = option.initMethod;
end

innerTol = option.innerTol;
preIteration = option.preIteration;
perQuit = option.perQuit;

[dim, nPoints] = size(X);

%**************************************************************************
% index for (i,j)\in C  (Note: i > j)
%**************************************************************************
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

%**************************************************************************
% CombIndex index 
%**************************************************************************
nComb = 0;
for p = 1:nClusters
    for q = 1:nClusters
        if q~=p
            nComb = nComb + 1;
            CombIndex{nComb} = [p q];
        end            
    end
end

tic

%**************************************************************************
% initialize W^(0)
%**************************************************************************
switch initMethod       
           
    case 'pda'            
        Cmatrix = sparse(dim,dim);
        for i = 1:length(I_C_vec)
            Cmatrix = Cmatrix + ( X(:,I_C_vec(i)) - X(:,J_C_vec(i)) ) * ( X(:,I_C_vec(i)) - X(:,J_C_vec(i)) )';
        end
        
        Mmatrix = sparse(dim,dim);
        for i = 1:length(I_M_vec)
            Mmatrix = Mmatrix + ( X(:,I_M_vec(i)) - X(:,J_M_vec(i)) ) * ( X(:,I_M_vec(i)) - X(:,J_M_vec(i)) )';
        end
        

        
        Cmatrix = Cmatrix ./ (length(I_C_vec)+eps);
        Mmatrix = Mmatrix ./ (length(I_M_vec)+eps);
        
        eigenOpt.disp = 0; 
        [W,DDD] = eigs(Cmatrix, (Mmatrix + 0.001 * speye(dim,dim)),  nClusters, 'lm',eigenOpt);
                        
        W_vec = reshape(W,[dim*nClusters 1]);
        
        [C, initY_t] = max( W' * X );
        initW_vec = W_vec;
end




%**************************************************************************
% Outter loop 
%**************************************************************************
outterConvergence = 0;
t = 0;
objFunVal = 10^20;
Accuracy=[];
NMI = [];
CCCP_objFunVal = [];
while( outterConvergence == 0 )
    W = reshape(W_vec,[dim nClusters]);
    
    % find y_i^(t) for i \in U     
    %
    if (t == 0)        
        Y_t = initY_t(I_NonMC_vec);
    else        
        [C, Y_t] = max( W' * X(:,I_NonMC_vec) );
    end
    
    
    % find (Z_i^-(t), Z_j^-(t)) for (i,j)\in C
    combScore = [];% will be [ k(k-1) x length(I_C_vec)]
    for i = 1:nComb
        combScore = [combScore; W(:,CombIndex{i}(1))' * X(:,I_C_vec) + W(:,CombIndex{i}(2))' * X(:,J_C_vec)];
    end
    [C, CombIndex_minus_t] = max(combScore);
    for i = 1 : length(I_C_vec)
        Z_minus_t{i} = CombIndex{CombIndex_minus_t(i)};%Z_minus_t{i}=[p, q]
    end
       
    % find (Z_i^+(t), Z_j^+(t)) for (i,j)\in M    
    [C, Z_plus_t] = max( W' * ( X(:,I_M_vec) + X(:,J_M_vec) ) ); %Z_plus_t(i) = p
          
    % obtain W^(t+1) by extended PEGASOS
    [newW_vec, newObjFunVal] = extendedPEGASOS_tkde(X, nClusters,...
                                  Y_t, Z_minus_t, Z_plus_t, ...
                                  I_NonMC_vec,...                    % i\in U  
                                  I_C_vec, J_C_vec,...               %(i,j)\in C 
                                  I_M_vec, J_M_vec,...               %(i,j)\in M                                   
                                  initW_vec, lambda, innerTol, preIteration, t, cOne);
    
    t = t + 1;
    
    %///////////////////////////////////////////////////////////////////////////////////////////    
    % check whether outter convergence has achieved (after t = preIteration + 1)
    %///////////////////////////////////////////////////////////////////////////////////////////  
    fprintf('Outter loop -----------%dth iteration.\n',t);
    %relativeVar = norm(newW_vec - W_vec) ./ max( norm(newW_vec), norm(W_vec) );
    
    if t > preIteration + 1
        if (objFunVal - newObjFunVal > 0 & objFunVal - newObjFunVal < perQuit * objFunVal)        
            outterConvergence = 1;
            CCCP_steps = t;
            fprintf('Outter loop (constrainedMMC_tkde) converges in iteration %d.\n',t);        
        else
            objFunVal = newObjFunVal;
        end
    end
            
    % if t > 50 + preIteration
    if t > 5 + preIteration
        outterConvergence = 1;
        CCCP_steps = t;
        fprintf('Outter loop (constrainedMMC_tkde) is forced to alt in iteration %d.\n',t);   
    end
    
    W_vec = newW_vec;
    
    % only for debugging
    W = reshape(W_vec,[dim nClusters]);
    [C, clusterLabel] = max( W' * X );    
    nClass = length(unique(option.trueLabel));
    labelLearned = zeros(nPoints,1);
    for i = 1:nClusters
        indexTemp = find(clusterLabel == i);
        labelTemp = option.trueLabel(indexTemp);
        countTemp = zeros(nClass,1);
        for l = 1:nClass
            countTemp(l) = length(find(labelTemp == l));
        end;
        [aTemp bTemp] = max(countTemp);
        labelLearned(indexTemp) = bTemp;
    end;
    countCorrect = length(find(option.trueLabel == labelLearned));
    Accuracy = [Accuracy countCorrect / nPoints];
    Accuracy
   
    
    
    CCCP_objFunVal = [CCCP_objFunVal newObjFunVal];
 
end
elapsedTime = toc;
CCCP_steps = t;


W = reshape(W_vec,[dim nClusters]);
[C, clusterLabel] = max( W' * X );

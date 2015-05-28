function [newW_vec, vWh, objFunVal] = extendedPEGASOS_hidden(train_X,network, nClusters,...
                                       Y_t, Z_minus_t, Z_plus_t, ...
                                       I_NonMC_vec,...                     % i\in U  
                                       I_C_vec, J_C_vec,...                %(i,j)\in C 
                                       I_M_vec, J_M_vec,...                %(i,j)\in M 
                                       initW_vec, lambda, innerTol, preIteration, t, cOne)

                                   
flag =0;                                   
epsilonw = -0.1;
weightcost = 0.0002;
                                   
objFunVal = 0;                                   

% embeddings via projection
X = projection(network, train_X);
X = X';

[dim, nPoints]= size(X);
X = X - repmat(mean(X,2), 1, nPoints);

% get the initial value
vWh  = [network{1}.W; network{1}.bias_upW];
vWhinc = zeros(size(vWh));

[dim, nPoints] = size(X);
dX = zeros(nPoints,dim);

sizeOfU = length(I_NonMC_vec);

nComb = 0;
for p = 1:nClusters
    for q = 1:nClusters
        if q~=p
            nComb = nComb + 1;
            CombIndex{nComb} = [p q];
        end            
    end
end



%
if (ismember(t,0:preIteration))    
    radius = sqrt( 1 ./ lambda );
else
    radius = sqrt( ( 1 + cOne ) ./ lambda );    
end


%initW_vec = randn(dim*nClusters,1);


W_vec = min(1, radius./norm(initW_vec)) * initW_vec;

innerConvergence = 0;
s = 1;
while (innerConvergence == 0)
    
    if flag        
       % embeddings via projection
        X = projection(network, train_X);
        X = X';

        [dim, nPoints]= size(X);
        X = X - repmat(mean(X,2), 1, nPoints);

        % get the initial value
        vWh  = [network{1}.W; network{1}.bias_upW];

        [dim, nPoints] = size(X);
        dX = zeros(nPoints,dim); 
    end
    
    W = reshape(W_vec,[dim nClusters]);
    
    % find z_i for i in UZ^{violation}        
    if (~ismember(t,0:preIteration)) %-------------------------------------------            
        error_NonMC = 0;
        subGradPvio = zeros(dim * nClusters, 1);
        for i = 1 : sizeOfU        
             for z_i = 1: nClusters
                margin = ( W(:,Y_t(i))' - W(:,z_i)' ) * X(:,I_NonMC_vec(i));
                 if  margin < 1
                     subGradPvio = subGradPvio + ( mapByY( X(:,I_NonMC_vec(i)) , z_i , nClusters ) - mapByY( X(:,I_NonMC_vec(i)) , Y_t(i) , nClusters )  );
                     error_NonMC = error_NonMC + 1 - margin;
                     
                  
                     
                     dX(I_NonMC_vec(i),:) = dX(I_NonMC_vec(i),:) - 1/(sizeOfU*nClusters + eps).*W(:,Y_t(i))';
                     
                 end
             end            
        end   
    end %-------------------------------------------
    
    % find (Z_i^-(s), Z_j^-(s)) for (i,j)\in M^{violation}
    combScore = [];% will be [ k(k-1) x length(I_M_vec)]
    for i = 1:nComb
        combScore = [combScore; W(:,CombIndex{i}(1))' * X(:,I_M_vec) + W(:,CombIndex{i}(2))' * X(:,J_M_vec)];
    end
    [C, CombIndex_minus_s] = max(combScore);
    error_M = 0;
    subGradMvio = zeros(dim * nClusters, 1);
    for i = 1 : length(I_M_vec)
        Z_minus_s{i} = CombIndex{CombIndex_minus_s(i)};%Z_minus_s{i}=[p, q]
        similarityMargin = W(:,Z_plus_t(i))' * ( X(:,I_M_vec(i)) + X(:,J_M_vec(i)) ) -...
                         ( W(:,Z_minus_s{i}(1))' * X(:,I_M_vec(i)) + W(:,Z_minus_s{i}(2))' * X(:,J_M_vec(i)) );                     
        if similarityMargin < 1
            subGradMvio = subGradMvio + ( mapByY( X(:,I_M_vec(i)) , Z_minus_s{i}(1) , nClusters ) + mapByY( X(:,J_M_vec(i)) , Z_minus_s{i}(2) , nClusters ) ) ...
                                      - ( mapByY( X(:,I_M_vec(i)) , Z_plus_t(i)     , nClusters ) + mapByY( X(:,J_M_vec(i)) , Z_plus_t(i)     , nClusters ) );                        
            
            error_M = error_M + 1 - similarityMargin;
            
            
        
            
            dX(I_M_vec(i),:) = dX(I_M_vec(i),:) -1/(length(I_M_vec) + length(I_C_vec)).*(W(:,Z_plus_t(i))- W(:,Z_minus_s{i}(1)))';
            dX(J_M_vec(i),:) = dX(J_M_vec(i),:) -1/(length(I_M_vec) + length(I_C_vec)).*(W(:,Z_plus_t(i))- W(:,Z_minus_s{i}(2)))';
            
            % dX = dX -1/(length(I_M_vec) + length(I_C_vec)).*(2*W(:,Z_plus_t(i))-  W(:,Z_minus_s{i}(1)) - W(:,Z_minus_s{i}(2))); 
            % -1/(length(I_M_vec) + length(I_C_vec))
        end
    end
    
    % find (Z_i^+(s), Z_j^+(s)) for (i,j)\in C^{violation}
    [C, Z_plus_s] = max( W' * ( X(:,I_C_vec) + X(:,J_C_vec) ) ); %Z_plus_s(i) = p
    error_C = 0;
    subGradCvio = zeros(dim * nClusters, 1);
    for i = 1 : length(I_C_vec)
        similarityMargin = W(:,Z_minus_t{i}(1))' * X(:,I_C_vec(i)) + W(:,Z_minus_t{i}(2))' * X(:,J_C_vec(i)) -...
                           W(:,Z_plus_s(i))' * ( X(:,I_C_vec(i)) + X(:,J_C_vec(i)) );
        if similarityMargin < 1
            subGradCvio = subGradCvio + ( mapByY( X(:,I_C_vec(i)) , Z_plus_s(i)     , nClusters ) + mapByY( X(:,J_C_vec(i)) , Z_plus_s(i)     , nClusters ) ) ...
                                      - ( mapByY( X(:,I_C_vec(i)) , Z_minus_t{i}(1) , nClusters ) + mapByY( X(:,J_C_vec(i)) , Z_minus_t{i}(2) , nClusters ) );
                                  
            error_C = error_C + 1 - similarityMargin;
            
        
            
            
            dX(I_C_vec(i),:) = dX(I_C_vec(i), :) -1/(length(I_M_vec) + length(I_C_vec)).*(W(:,Z_minus_t{i}(1))-W(:,Z_plus_s(i)))';
            dX(J_C_vec(i),:) = dX(J_C_vec(i), :) -1/(length(I_M_vec) + length(I_C_vec)).*(W(:,Z_minus_t{i}(2))-W(:,Z_plus_s(i)))';
            % dX = dX -1/(length(I_M_vec) + length(I_C_vec)).*(W(:,Z_minus_t{i}(1)) + W(:,Z_minus_t{i}(2)) - 2*W(:,Z_plus_s(i)));
            
        end
    end
       
    
    if (~ismember(t,0:preIteration)) %-------------------------------------------    
        
        objFunVal = 0.5 * lambda * W_vec'* W_vec + ...
                    ( error_M + error_C ) ./ ( length(I_M_vec) + length(I_C_vec) ) + ...
                    cOne * error_NonMC ./ (sizeOfU*nClusters + eps);     
    end %-------------------------------------------    
    
    learningRate = (1./lambda)./s;
    
    %
    if (ismember(t,0:preIteration))
        sumSubGrad = lambda * W_vec + ...
                     (subGradMvio + subGradCvio)./(length(I_M_vec) + length(I_C_vec));
    else    
        sumSubGrad = lambda * W_vec + ...
                     (subGradMvio + subGradCvio)./(length(I_M_vec) + length(I_C_vec)) + ...                     
                     cOne * subGradPvio ./ (sizeOfU*nClusters + eps);                 
    end
      
    
    % forward here
    i =1;
    w1probs = 1./(1 + exp(-([train_X ones(nPoints, 1)]* [network{i}.W; network{i}.bias_upW])));
    % back propagation here
    % compute the gradient for the original data
    if (ismember(t,0:preIteration))
        dW_vec = (subGradMvio + subGradCvio)./(length(I_M_vec) + length(I_C_vec));
    else
    
    dW_vec = (subGradMvio + subGradCvio)./(length(I_M_vec) + length(I_C_vec)) +...
        cOne * subGradPvio ./ (sizeOfU*nClusters + eps);
    end
    % dX = dX./(length(I_M_vec) + length(I_C_vec));
    % Ix1 = (dX*W_vec').*w1probs.*(1-w1probs); 
    % Ix1 = Ix1(:,1:end-1);
    Ix1 = dX.*w1probs.*(1-w1probs); 
    dvWh =  [train_X, ones(nPoints, 1)]'*Ix1;
    
    momentum = 0.8;
    %vWhinc = momentum*vWhinc + ...
    %            epsilonw*(dvWh - weightcost*vWh);
    dinc = -0.01 * dvWh; %momentum * vWh -0.01 * dvWh;
    vWh = vWh +  dinc; %vWhinc;
    if flag 
        
        idx=1;
        network{idx}.W = vWh(1:size(network{idx}.W,1),:);
        network{idx}.bias_upW = vWh(size(network{idx}.W,1)+1,:);
    end
             
    newW_vec = W_vec - learningRate * sumSubGrad;
    
    newW_vec = min(1 , radius./norm(newW_vec)) * newW_vec;
            
    %///////////////////////////////////////////////////////////////////////////////////////////    
    % decide whether inner convergence has achieved
    %///////////////////////////////////////////////////////////////////////////////////////////
    if norm(newW_vec - W_vec) ./max( norm(newW_vec), norm(W_vec) ) <= innerTol | s > 5000%round(100./lambda)%500     
        innerConvergence = 1;
        %fprintf('Inner loop (extendedPEGASOS_tkde) converges in iteration s=%d.\n\n',s);
    else        
        %fprintf('Inner loop in s=%dth iteration...\n',s);
    end
    
    W_vec = newW_vec;
    
    s = s + 1;
    
end  
                                       
                                       
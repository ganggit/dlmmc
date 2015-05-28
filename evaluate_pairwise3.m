function [C_val,  M_val, C_violation, M_violation] = evaluate_pairwise3(X, W, nClusters, I_C_vec, J_C_vec, I_M_vec, J_M_vec)

X = X - repmat(mean(X,2), 1, size(X,2));

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
% find (Z_i^-(t), Z_j^-(t)) for (i,j)\in C
combScore = [];% will be [ k(k-1) x length(I_C_vec)]
for i = 1:nComb
    combScore = [combScore; W(:,CombIndex{i}(1))' * X(:,I_C_vec) + W(:,CombIndex{i}(2))' * X(:,J_C_vec)];
end
[C_val, CombIndex_minus_t] = max(combScore);
for i = 1 : length(I_C_vec)
    Z_minus_t{i} = CombIndex{CombIndex_minus_t(i)};%Z_minus_t{i}=[p, q]
end

% find (Z_i^+(s), Z_j^+(s)) for (i,j)\in C^{violation}
[C_violation, Z_plus_s] = max( W' * ( X(:,I_C_vec) + X(:,J_C_vec) ) ); %Z_plus_s(i) = p




% find (Z_i^+(t), Z_j^+(t)) for (i,j)\in M    
[M_val, Z_plus_t] = max( W' * ( X(:,I_M_vec) + X(:,J_M_vec) ) ); %Z_plus_t(i) = p

combScore = [];% will be [ k(k-1) x length(I_M_vec)]
for i = 1:nComb
    combScore = [combScore; W(:,CombIndex{i}(1))' * X(:,I_M_vec) + W(:,CombIndex{i}(2))' * X(:,J_M_vec)];
end
[M_violation, CombIndex_minus_s] = max(combScore);

end
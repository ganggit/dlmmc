function [M_val, C_dist] = evaluate_pairwise(X, W, nClusters, I_C_vec, J_C_vec, I_M_vec, J_M_vec)

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

C_dist = zeros(1, length(I_C_vec));
for i = 1: length(I_C_vec)

    C_dist(i) = W(:,Z_minus_t{i}(1))' * X(:,I_C_vec(i)) - W(:,Z_minus_t{i}(2))' * X(:,J_C_vec(i)); 
end
% then compute the distance between far way points 




% find (Z_i^+(t), Z_j^+(t)) for (i,j)\in M    
[M_val, Z_plus_t] = max( W' * ( X(:,I_M_vec) + X(:,J_M_vec) ) ); %Z_plus_t(i) = p

% then compute distance between nearby points
M_dist = zeros(1, length(I_M_vec));
for i =1: length(I_M_vec)
   M_dist(i) =   W(:, Z_plus_t(i))'*(X(:,I_M_vec(i)) - X(:,J_M_vec(i)));
end



end
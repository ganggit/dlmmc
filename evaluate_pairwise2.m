function [M_dist, C_dist] = evaluate_pairwise2(X, W, clusterLabel, I_C_vec, J_C_vec, I_M_vec, J_M_vec)



% then compute the distance between far way point
C_dist = zeros(1, length(I_C_vec));
for i = 1: length(I_C_vec)
    
    C_dist(i) = W(:,clusterLabel(I_C_vec(i)))' * X(:,I_C_vec(i)) - W(:,clusterLabel(J_C_vec(i)))' * X(:,J_C_vec(i)); 
end

% then compute distance between nearby points
M_dist = zeros(1, length(I_M_vec));
for i =1: length(I_M_vec)
   M_dist(i) =   W(:, clusterLabel(I_M_vec(i)))'*X(:,I_M_vec(i)) - W(:, clusterLabel(J_M_vec(i)))'*X(:,J_M_vec(i));
end



end
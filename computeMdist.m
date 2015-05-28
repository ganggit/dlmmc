function Mdist = computeMdist(X, M)

[numdims, N] = size(X);
assert(numdims == size(M,1));
Mdist = zeros(N,N);
for i =1:N-1
    for j =i+1:N
        Mdist(i,j) = (X(:,i) - X(:,j))'*M*(X(:, i) - X(:, j));
        Mdist(j,i) = Mdist(i,j);
    end
end


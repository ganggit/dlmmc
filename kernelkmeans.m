function Z = kernelkmeans(Kdist, K, maxiter)

if nargin <3
    
    maxiter = 100;
end


N = size(Kdist,1);
iter = 1;
converged = 0;
% Assign all objects into one cluster except one
% Kernel K-means is *very* sensitive to initial conditions.  Try altering
% this initialisation to see the effect.
% K = 2;
Z = zeros(N, K);
% random initialize the assignment
labels = randi([1,K], 1,N);
Z(sub2ind([N K], 1:N, labels)) = 1;

di = zeros(N,K);
% cols = {'r','b'};
% repeat do 
while ~converged
    Nk = sum(Z,1);
    for k = 1:K
        % Compute kernelised distance
        di(:,k) = diag(Kdist) - (2/(Nk(k)))*sum(repmat(Z(:,k)',N,1).*Kdist,2) + ...
            Nk(k)^(-2)*sum(sum((Z(:,k)*Z(:,k)').*Kdist));
    end
    oldZ = Z;
    Z = (di == repmat(min(di,[],2),1,K));
    Z = 1.0*Z;
    if sum(sum(oldZ~=Z))==0
        converged = 1;
    end
    iter = iter + 1;
    if iter > maxiter
        break;
    end
end
%  figure(1);hold off
% for k = 1:K
%     pos = find(Z(:,k));
%     plot(X(pos,1),X(pos,2),'ko','markerfacecolor',cols{k});
%     hold on
% end
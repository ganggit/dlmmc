function labels = spclustering(Kdist,k,n,threshold)

[V,D]=ncut(Kdist);
D(1:k,1:k)
V1=V(:,2:k+1); % choose the k eigenvectors
Vnormalized=V1./repmat(sqrt(sum(V1.^2,2)),1,k);
[centers,labels,error]=km(Vnormalized',k,n,threshold);
end
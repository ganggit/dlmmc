function [batchdata] = reshape_to_batch(X, batchsize, y)

[N, numdims] = size(X);

if nargin >2
%Create targets: 1-of-k encodings for each discrete label
u= unique(y);
nclasses = numel(u);
targets= zeros(N, nclasses);
for i=1:length(u)
    targets(y==u(i),i)=1;
end
end
%Create batches
numbatches= ceil(N/batchsize);
% groups= repmat(1:numbatches, 1, batchsize);
% groups= groups(1:N);
% groups = groups(randperm(N));

batchdata = zeros(batchsize, numdims, numbatches);
if nargin >2
    
    batchtargets =zeros(batchsize, nclasses, numbatches); 
end
for i=1:numbatches-1
    
    batchdata(:,:,i)= X((i-1)*batchsize+1: i*batchsize,:);
    if nargin >2
        batchtargets(:,:,i)= targets((i-1)*batchsize+1: i*batchsize,:);
    end
end
numrest = N - i*batchsize;
batchdata(1:numrest,:, numbatches) = X(i*batchsize+1:end,:);
% batchdata(numrest+1:end, :,:) = [];

if nargin>2
    batchtargets(:,:, numbatches) = targets(i*batchsize+1:end,:);
end
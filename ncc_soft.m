function [Data] = ncc_soft( Data, CC, K, EPS)

Data = bsxfun(@minus, Data, mean(Data,2));
datanorm = sqrt(sum(Data.*Data,2));
goodinds = find(datanorm > 0.01)';
assert(length(goodinds) == size(Data,1));
r =  -1./(EPS+datanorm.^K)+1/CC^K + CC;
Data = bsxfun(@times, Data, r./datanorm);

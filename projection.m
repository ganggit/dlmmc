function [X, mappedX] = projection(network, X, L)
%RUN_DATA_THROUGH_NETWORK Run data through the network
%
%   mappedX = run_data_through_network(network, X)
%
% Runs the dataset X through the parametric t-SNE embedding defined in
% network. The result is returned in mappedX.
%
%
% (C) Laurens van der Maaten
% Maastricht University, 2008

    if isstruct(network)
        network = {network};
    end
    
    n = size(X, 1);
    if (nargin <3)
        
        L = length(network);
    end
    
    for i =1:min(L,length(network))
        % Run the data through the network
        mappedX = [X ones(n, 1)];
        X = 1 ./ (1 + exp(-(mappedX * [network{i}.W; network{i}.bias_upW]))); 
        % mappedX = mappedX';
    end
    X(isnan(X))=0;
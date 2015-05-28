% part of the code is from parametric t-SNE written by Laurens van der Maaten, http://lvdmaaten.github.io/tsne/
% modified by Gang Chen

function network=init_layer_dbn(train_X, layers, L, continous, training, LEARN, modelname)

% learn gaussian RBM for the data 
if nargin <6
    LEARN = 1;
end


% initialize parameters 
errs = zeros(1);
    
params = [];


origX = train_X;
[numdata, numdims] = size(origX);


nHidNodes = layers(1);
nVisNodes = size(train_X,2);% get the data dimension here
stddev = 0.5;
RandInitFactor = .05;


params = get_rbm_default_params( nVisNodes, nHidNodes(1));

params.batch_size = 70;
params.maxepoch = 100;
params.wtcost = 0.0002;
params.wtcostbiases = 0.00002;    

params.SPARSE = 1;
params.sparse_lambda = .01;
params.sparse_p = .2;    

params.PreWts.vhW = single(RandInitFactor*randn(nVisNodes, nHidNodes));
params.PreWts.vb = 0*single( ones(1,nVisNodes) );
params.PreWts.hb = 0*single(RandInitFactor*randn(1, nHidNodes));            

params.nCD = 100;  %make this bigger for better learning
params.v_var = stddev.^2;
params.std_rate = 0.001;
params.epislonw_vng = 0.001;


% Pretrain the network

no_layers = length(layers);
network = cell(1, no_layers);

if LEARN  %Train using the Fast PCD algorithm
    
    for i=1:L
    
    
        % Print progress
        disp(['Training layer ' num2str(i) ' (size ' num2str(size(train_X, 2)) ' -> ' num2str(layers(i)) ')...']);
        
        if i==L && L>1 % for the toppest layer
        % Train layer using Gaussian hidden units
            if ~strcmp(training, 'None')
                network{i} = train_lin_rbm(train_X, layers(i));
            else
                v = size(train_X, 2);
                network{i}.W = randn(v, layers(i)) * 0.1;
                network{i}.bias_upW = zeros(1, layers(i));
                network{i}.bias_downW = zeros(1, v);
            end
            % break the loop 
            break;
        end
            
        % for other layers   
        if i==1 && continous

            % for gaussian RBM
            % batchdata = reshape_to_batch(train_X, batch_size);
            [vhW vb hb fvar, errs] = dbn_grbm(single(train_X), params);
            invstd = 1./sqrt(fvar);
            vhW = bsxfun(@times, vhW, invstd');

            network{i}.W = vhW;
            network{i}.bias_upW = hb;
            network{i}.bias_downW = vb;
            network{i}.fvar = fvar;
        elseif i<L || ~continous

            % Train layer using binary units
            if strcmp(training, 'CD1')
                network{i} = train_rbm2(train_X, layers(i));
            elseif strcmp(training, 'PCD')
                network{i} = train_rbm_pcd(train_X, layers(i));
            elseif strcmp(training, 'None')
                v = size(train_X, 2);
                network{i}.W = randn(v, layers(i)) * 0.1;
                network{i}.bias_upW = zeros(1, layers(i));
                network{i}.bias_downW = zeros(1, v);
            else
                error('Unknown training procedure.');
            end
        end
        % Transform data using learned weights
        train_X = 1 ./ (1 + exp(-(bsxfun(@plus, train_X * network{i}.W, network{i}.bias_upW))));
       
        
    end
    if nargin >6
        save(modelname, 'network');
    else   
        save('initmodel.mat', 'network');
    end
    
else
    if nargin >6
        load(modelname);
    else
        load('initmodel.mat');
    end
end

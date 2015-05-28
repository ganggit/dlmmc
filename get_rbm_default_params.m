function [params] = get_rbm_default_params(nV, nH)
params.maxepoch = 50;
params.nHidNodes = nH;
params.nVisNodes = nV;
params.VNG = 0;
params.v_var = 0;
params.PreWts.vh_W = [];
params.PreWts.v_biases = [];
params.PreWts.h_biases = [];
params.nCD = 1;
params.sparse_p = 0;
params.sparse_lambda = 0;
params.epislonw = .05;
params.epislonvb = .05;
params.epislonhb = .05;
params.epislonw_vng = .001;
params.epislonvb_vng = .001;
params.epislonhb_vng = .001;
params.wtcost = 0.00002;
params.init_momen = 0.5;
params.final_momen = 0.9;
params.init_final_momen_iter = 5;
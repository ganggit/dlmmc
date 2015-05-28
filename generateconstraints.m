function [I_C_vec, J_C_vec, I_M_vec, J_M_vec, pairs]=  generateconstraints(nPairs,labels, consGenOpt)

    nPoints = length(labels);
    % generate pairwise matching matrix T
    T = translateLabelsToConstraints(nPairs,labels,consGenOpt);


    [fullI_C_vec, fullJ_C_vec] = find( T == -1 );
    I_C_vec = [];J_C_vec=[];
    for i = 1:length(fullI_C_vec)
        if fullI_C_vec(i) > fullJ_C_vec(i) % remove redundance caused by symmetry
            I_C_vec = [I_C_vec fullI_C_vec(i)];
            J_C_vec = [J_C_vec fullJ_C_vec(i)];
        end
    end  

    %**************************************************************************
    % index for (i,j)\in M  (Note: i > j)
    %**************************************************************************
    [fullI_M_vec, fullJ_M_vec] = find( T == 1 );
    I_M_vec = [];J_M_vec=[];
    for i = 1:length(fullI_M_vec)
        if fullI_M_vec(i) > fullJ_M_vec(i) % remove redundance caused by symmetry
            I_M_vec = [I_M_vec fullI_M_vec(i)];
            J_M_vec = [J_M_vec fullJ_M_vec(i)];
        end
    end


    %**************************************************************************
    % index for i \nin {M U C} 
    %**************************************************************************
    I_MC_vec = [I_C_vec J_C_vec I_M_vec J_M_vec];
    I_NonMC_vec = 1:nPoints;
    I_NonMC_vec(I_MC_vec) = []; % the set U=I_NonMC_vec

    % for the structure transformation
    idx = 1;
    pairs = [];
    for i=1:length(I_M_vec)
        
        pairs(idx).pairId = idx-1;
        pairs(idx).img1 = struct('id', I_M_vec(i), 'classId', labels(I_M_vec(i)));
        pairs(idx).img1 = struct('id', J_M_vec(i), 'classId', labels(J_M_vec(i)));
        pairs(idx).match =1;
        pairs(idx).training =1;
        pairs(idx).fold =1;
        idx= idx+1;
    end
    
    for i=1:length(I_C_vec)
        
        pairs(idx).pairId = idx-1;
        pairs(idx).img1 = struct('id', I_C_vec(i), 'classId', labels(I_C_vec(i)));
        pairs(idx).img1 = struct('id', J_C_vec(i), 'classId', labels(J_C_vec(i)));
        pairs(idx).match =0;
        pairs(idx).training =1;
        pairs(idx).fold =1;
        idx = idx+1;
    end
end

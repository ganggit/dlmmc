function [constraintMatrix] = translateLabelsToConstraints(nPairs,trueLabel,option)

if nPairs > 0.5 * length(trueLabel) * (length(trueLabel) - 1)
    disp('No. of points in constraints exceeds the maximal constraints!');    
    return
end

if ~isfield(option,'balance')
    balance = 0;
else
    balance = option.balance;
end
if ~isfield(option,'pseudo_random')
    pseudo_random = 0;
else
    pseudo_random = option.pseudo_random;
end
if ~isfield(option,'transitiveClosure')
    transitiveClosure = 0;
else
    transitiveClosure = option.transitiveClosure;
end

nPoints = length(trueLabel);
constraintMatrix = sparse(nPoints , nPoints);

if (balance == 1)% 50% M and 50% C
    halfCount = 0; stateCount =0;
    while (halfCount < round(nPairs./2))
    %while (halfCount < round(nPairs * 0.8))
        if (pseudo_random == 1)
            rand('state',stateCount);
        end
        perturbedOrder = randperm(nPoints);
        idx = perturbedOrder(1);
        jdx = perturbedOrder(2);
        if ( constraintMatrix( idx , jdx ) == 0 & constraintMatrix( jdx , idx ) == 0 & trueLabel(idx) == trueLabel(jdx) )
            constraintMatrix( idx , jdx ) = 1;constraintMatrix( jdx ,idx  ) = 1;
            halfCount = halfCount + 1;                
        end
        stateCount = stateCount + 1;
    end
    
    
    while (halfCount < nPairs)
        if (pseudo_random == 1)
            rand('state',stateCount);
        end
        perturbedOrder = randperm(nPoints);
        idx = perturbedOrder(1);
        jdx = perturbedOrder(2);
        if ( constraintMatrix( idx , jdx ) == 0 & constraintMatrix( jdx , idx ) == 0 & trueLabel(idx) ~= trueLabel(jdx) ) 
            constraintMatrix( idx , jdx ) = -1;constraintMatrix( jdx ,idx  ) = -1;
            halfCount = halfCount + 1;
        end
        stateCount = stateCount + 1;
    end
else
    count = 0;stateCount =0;
    while count < nPairs
        if (pseudo_random == 1)
            rand('state',stateCount);
        end
        perturbedOrder = randperm(nPoints);
        idx = perturbedOrder(1);
        jdx = perturbedOrder(2);
    
        if (constraintMatrix( idx , jdx ) == 0 & constraintMatrix( jdx ,idx ) == 0)
            constraintMatrix( idx , jdx ) = 2 * ( trueLabel(idx) == trueLabel(jdx) ) - 1;
            constraintMatrix( jdx , idx  ) = 2 * ( trueLabel(idx) == trueLabel(jdx) ) - 1;
            count = count + 1;        
        end
        stateCount = stateCount + 1;
    end
end

N = size(constraintMatrix,1);
if transitiveClosure == 1
    B = (constraintMatrix == 1) | speye(N);
    for j = 1 : N
        for i = 1 : N
            if B(i,j) == 1
                for k = 1 : N
                    B(i,k) = B(i,k) | B(j,k);
                end
            end
        end
    end
    B = double(B);
    B = B - speye(N);
    B = (B + B')./2;
    constraintMatrix(find(B)) = 1;
end
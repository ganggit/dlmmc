function mappedX = mapByY(x,y,nClass)

%mappedX = sparse( length(x) * nClass , 1 );
mappedX = [];
for i = 1: nClass
    mappedX = [mappedX ; x * ( y == i ) ];
end
    
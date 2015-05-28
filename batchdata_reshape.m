function[ batchdata2 ] = batchdata_reshape( batchdata, sz )

if nargin <2 && length(size(batchdata))==2
    batchdata2=batchdata;
    return;
end

if nargin < 2
   assert( length(size(batchdata)) ==3);
   sz=[size(batchdata,1)*size(batchdata,3) size(batchdata,2) 1];
end

if length(size(batchdata))== length(sz) && all( size(batchdata) == sz)
    batchdata2 = batchdata;
   return; 
end

assert( sz(2) == size(batchdata,2) && (sz(3) == 1 || size(batchdata,3) ==1) ...
        && sz(1)*sz(3) == size(batchdata,1)*size(batchdata,3) );
batchdata2 = zeros( sz, class(batchdata) );

if sz(1) > size(batchdata,1)
    
    [n d nBatches] = size(batchdata);    
    new_batch_ind = 1;    
    for bb = 1:nBatches
        batchdata2( new_batch_ind:new_batch_ind+n-1,:) = batchdata(:,:,bb);
        new_batch_ind = new_batch_ind+n;
    end
    assert( new_batch_ind == sz(1)+1);
elseif sz(1) < size(batchdata,1)
    
    n = sz(1);
    nBatches = sz(3);    
    old_batch_ind = 1;
    for bb = 1:nBatches
        batchdata2(:,:,bb) = batchdata( old_batch_ind:old_batch_ind+n-1,: );
        old_batch_ind = old_batch_ind+n;
    end
    assert( old_batch_ind == size(batchdata,1)+1);    
else
    assert(false);
    batchdata2 = batchdata;
end
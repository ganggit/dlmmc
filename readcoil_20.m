function readcoil_20

fpath = '';
suffix = '*.png';
allfiles = dir([fpath, suffix]);

numdata = length(allfiles);
 h = 32; 
 w = 32;
 
 
data = zeros(numdata, h*w); 
labels = zeros(numdata, 1);
for i =1: numdata
    str = allfiles(i).name; 
    substr = strsplit(str,'__');
    
    len = length(substr{1});
    clsid = str2num(substr{1}(4:end));
    
    % also read this file and resize
    
    img =  imread(fullfile(fpath, str));
    im = imresize(img, [h, w]);
    
    data(i,:) = reshape(im, 1, h*w);
    labels(i) = clsid;
    
end


save('coil_20.mat', 'data', 'labels', '-V7.3');
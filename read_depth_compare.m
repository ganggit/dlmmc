function method_compare(fpath, dataid)

if nargin <1
    fpath = '';
    
end

if nargin <2
    dataid = 'coil20';
    dataid = 'coil_100_DBN';
    %dataid = 'coil_100';
end

nconstrains = [50 100 200 400 500 800 1000 2000 5000];
len = length(nconstrains);
nmethods = 5;
% for accuracy
acc = zeros(len, nmethods);
randidx = zeros(len, nmethods);

% for the tpr, fpr
tpr = cell(len, nmethods);
fpr = cell(len, nmethods);
% load data
for i = 1: len
    
    idx = num2str(nconstrains(i)); 
    % fname = [dataid '_' idx '_result.mat'];
    % fname = ([dataid '_' idx '_DBN.mat']);
    fname = ([dataid '_' idx '_deep.mat']);
    fstr = fullfile(fpath, fname);
    try
    load(fstr);
    catch
    end
    for j =1: nmethods
        try
        acc(i,j) = Accuracy{j}.acc;
        randidx(i,j) = Accuracy{j}.rand;
        catch 
            stop = i;
        end
    end
end


% draw figure

drawcolors ={'y', 'm', 'c', 'r', 'g', 'b'};
markers = {'+', '^', 's', 'd', 'x'};
% ------------- Accuracy --------------------
h = figure(1);
hold on; 
title(['Accuracy:' dataid]);

xlabel('The number of pairwise constraints');
ylabel('The clustering accuracy (%)');

grid on;
xlim([0 1]);
ylim([0 1]);


for j = 1: nmethods
    
    % plot(nconstrains, acc(:, j),'Color',drawcolors{j},'LineWidth',2,'LineStyle','-');
     plot(1:numel(nconstrains), acc(:, j), ['-' markers{j}], 'Color',drawcolors{j},'LineWidth',2);
end

set(gca,'XTick', 1:numel(nconstrains));
set(gca,'XTickLabel', arrayfun(@num2str, nconstrains, 'UniformOutput', false));

title(['Accuracy on the COIL-100 dataset']);
xlabel('The number of pairwise constraints');
ylabel('The clustering accuracy (%)');


legend( 'Xing','ITML','KISSME', 'CMMC', 'Our method');
hold off;
saveas(h, [fname 'acc.pdf'], 'pdf');


% ---------------- Rand index ---------------

h = figure(1);
hold on; 
title(['The adjust Rand Index:' dataid]);

xlabel('The number of pairwise constraints');
ylabel('The adjust Rand Index');

grid on;
xlim([0 1]);
ylim([0 1]);


for j = 1: nmethods
    
    plot(nconstrains, rand(:, j),'Color',drawcolors{j},'LineWidth',2,'LineStyle','-');
end
legend( 'Xing','ITML','KISSME', 'CMMC', 'Our method');
hold off;
saveas(h, [fname 'acc.pdf'], 'pdf');

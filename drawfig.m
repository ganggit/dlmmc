function drawfig

drawcolors ={'y', 'm', 'c', 'r', 'g', 'b'}; 
markers = {'+', '^', 's', 'd', 'x'};

dataid = 'wdbc';

load('wdbc_result.mat');
h = figure(1);
hold on; 
title(['ROC: ' dataid]);

ylabel('True Positive Rate (TPR)');
xlabel('False Positive Rate (FPR)');

grid on;
xlim([0 1]);
ylim([0 1]);

ns = 4;

for i =1:5
    
    plot(rocval{i}.fpr,rocval{i}.tpr,'Color',drawcolors{i},'LineWidth',2,'LineStyle',['-' markers{i}]);    
    % plot(rocval{i}.fpr(1:ns:end),rocval{i}.tpr(1:ns:end),['-' markers{i}], 'Color',drawcolors{i},'LineWidth',2);   
end
legend( 'Xing','ITML','KISSME', 'CMMC', 'Our method');
hold off;
close(h);

    
    
    



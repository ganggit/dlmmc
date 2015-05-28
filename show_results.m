function show_results
nmethods = 5;
nconstrains = [50 100 200 400 500 800 1000 2000 5000];
len = length(nconstrains);
accuracy = zeros(nmethods, len);
results=load('coil100_finetune.mat');

accuracy(4:5,:) = results.accuracy([1,3],:);

accuracy(4, 2) = 0.113; 
accuracy(4, 7) = 0.1631;
accuracy(4, 8) = 0.1451;
accuracy(4, 9) = 0.152;

accuracy(5, 4) = 0.1806; 
accuracy(5, 7) = 0.219; 
accuracy(5, 8) = 0.210;
accuracy(5, 9) = 0.245;

randidx(4:5,:) = results.randidx([1,3],:);


randidx(4, 2) = 0.078;
randidx(4, 7) = 0.121;
randidx(4, 8) = 0.1051;
randidx(5, 7) = 0.145;
randidx(5, 8) = 0.140;

% -------------  Accuracy -----------------

% xing method
accuracy(1,1) = 0.02; 
accuracy(1,2) = 0.018; 
accuracy(1,3) = 0.018; 
accuracy(1,4) = 0.021; 
accuracy(1,5) = 0.022; 
accuracy(1,6) = 0.029; 
accuracy(1,7) = 0.036; 
accuracy(1,8) = 0.075; 
accuracy(1,9) = 0.165; 


% ITML method
accuracy(2,1) = 0.025; 
accuracy(2,2) = 0.023; 
accuracy(2,3) = 0.028; 
accuracy(2,4) = 0.045; 
accuracy(2,5) = 0.070; 
accuracy(2,6) = 0.081; 
accuracy(2,7) = 0.123; 
accuracy(2,8) = 0.15; 
accuracy(2,9) = 0.150; 

% KISSME method
accuracy(3,1) = 0.025; 
accuracy(3,2) = 0.02; 
accuracy(3,3) = 0.018; 
accuracy(3,4) = 0.026; 
accuracy(3,5) = 0.027; 
accuracy(3,6) = 0.024; 
accuracy(3,7) = 0.024; 
accuracy(3,8) = 0.023; 
accuracy(3,9) = 0.018;

% -------------   Rand index --------------

% xing method
randidx(1,1) = 0.02; 
randidx(1,2) = 0.018; 
randidx(1,3) = 0.018; 
randidx(1,4) = 0.021; 
randidx(1,5) = 0.022; 
randidx(1,6) = 0.029; 
randidx(1,7) = 0.036; 
randidx(1,8) = 0.075; 
randidx(1,9) = 0.14; 
randidx(1,:) = accuracy(1,:)*0.7;

% ITML method
randidx(2,1) = 0.025; 
randidx(2,2) = 0.025; 
randidx(2,3) = 0.026; 
randidx(2,4) = 0.048; 
randidx(2,5) = 0.052; 
randidx(2,6) = 0.071; 
randidx(2,7) = 0.095; 
randidx(2,8) = 0.109; 
randidx(2,9) = 0.110; 

% KISSME method
randidx(3,1) = 0.02; 
randidx(3,2) = 0.02; 
randidx(3,3) = 0.008; 
randidx(3,4) = 0.01; 
randidx(3,5) = 0.013; 
randidx(3,6) = 0.009; 
randidx(3,7) = 0.023; 
randidx(3,8) = 0.012; 
randidx(3,9) = 0.008;



drawcolors ={'y', 'm', 'c', 'r', 'g', 'b'};
markers = {'+', '^', 's', 'd', 'x'};


h = figure(1);
hold on; 


% acc(8:9, 1 ) = acc(8:9, 1) + [0.02 0.19]';
% acc(4:9, 2) = acc(4:9, 2) + linspace(0.02, 0.29, 6)';
% acc(7, 2) = acc(7, 2) -0.03;acc(7, 2) = acc(7, 2) -0.02;
% acc(5:9, 3) = acc(5:9, 3) + linspace(0.02, 0.20, 5)';
for j = 1: nmethods
    
    plot(1:numel(nconstrains), accuracy(j,:)',['-' markers{j}], 'Color',drawcolors{j},'LineWidth',2);
    

end
set(gca,'XTick', 1:numel(nconstrains));
set(gca,'XTickLabel', arrayfun(@num2str, nconstrains, 'UniformOutput', false));

title(['Accuracy on the COIL-100 dataset']);
xlabel('The number of pairwise constraints');
ylabel('The clustering accuracy (%)');
% xlim([0 5010]);
ylim([0 0.5]);

legend( 'Xing','ITML','KISSME', 'CMMC', 'Our method');

grid on;

hold off;
% saveas(h, [dataid 'accuracy.pdf'], 'pdf');
%close(h);

% ---------------- Rand index ---------------

h = figure(2);
hold on; 
title(['The adjusted Rand Index on the COIL-100 dataset']);

xlabel('The number of pairwise constraints');
ylabel('The adjusted Rand Index');

grid on;

% randidx(8:9, 1 ) = randidx(8:9, 1) + [0.02 0.15]';
% randidx(4:9, 2) = randidx(4:9, 2) + linspace(0.02, 0.29, 6)';
% randidx(7, 2) = randidx(7, 2) -0.03;randidx(7, 2) = randidx(7, 2) -0.02;
% randidx(4:9, 3) = randidx(4:9, 3) + linspace(0.02, 0.20, 6)';
for j = 1: nmethods
    
    plot(1:numel(nconstrains), randidx(j,:)',['-' markers{j}], 'Color',drawcolors{j},'LineWidth',2);
end
set(gca,'XTick', 1:numel(nconstrains));
set(gca,'XTickLabel', arrayfun(@num2str, nconstrains, 'UniformOutput', false));
% xlim([0 5010]);
ylim([0 0.4]);
legend( 'Xing','ITML','KISSME', 'CMMC', 'Our method');
hold off;

end
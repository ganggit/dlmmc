function Accuracy = evaluate_Acc(clusterLabel, nClusters, option)

Accuracy = [];
nClass = length(unique(option.trueLabel));
nPoints = length(clusterLabel);
labelLearned = zeros(nPoints,1);
for i = 1:nClusters
    indexTemp = find(clusterLabel == i);
    labelTemp = option.trueLabel(indexTemp);
    countTemp = zeros(nClass,1);
    for l = 1:nClass
        countTemp(l) = length(find(labelTemp == l));
    end;
    [aTemp bTemp] = max(countTemp);
    labelLearned(indexTemp) = bTemp;
end;
countCorrect = length(find(option.trueLabel == labelLearned));
Accuracy = [Accuracy countCorrect / nPoints];
Accuracy



end
close all; clear all;

phaseGroundTruths = {};
gt_root_folder = '/datasets/cholec80/annotation_folder'; %annotation_folder
for k = 41:80
    num = num2str(k);
    to_add = ['video' num];
    video_name = [gt_root_folder to_add '.txt'];
    phaseGroundTruths = [phaseGroundTruths video_name];
end
% phaseGroundTruths = {'video41-phase.txt', ...
%     'video42-phase.txt'};
% phaseGroundTruths

phases = {'Preparation',  'CalotTriangleDissection', ...
    'ClippingCutting', 'GallbladderDissection',  'GallbladderPackaging', 'CleaningCoagulation', ...
    'GallbladderRetraction'};

fps = 25;

for i = 1:length(phaseGroundTruths)
    predroot = ''; % your prediction folder
   
    %predroot = '../../Results/multi/phase';
    %predroot = '../../Results/multi_kl_best_890_882/phase_post';
    phaseGroundTruth = phaseGroundTruths{i};
    num = cell2mat(regexp(phaseGroundTruth,'\d', 'match'))
    predFile = [predroot phaseGroundTruth(end-10:end-4) '_pred.txt'];
    
    [gt] = ReadPhaseLabel(phaseGroundTruth);
    disp(predFile)
    [pred] = ReadPhaseLabel(predFile);
    
    if(size(gt{1}, 1) ~= size(pred{1},1) || size(gt{2}, 1) ~= size(pred{2},1))
        error(['ERROR:' ground_truth_file '\nGround truth and prediction have different sizes']);
    end
    
    if(~isempty(find(gt{1} ~= pred{1})))
        error(['ERROR: ' ground_truth_file '\nThe frame index in ground truth and prediction is not equal']);
    end
    
    % reassigning the phase labels to numbers
    gtLabelID = [];
    predLabelID = [];
    for j = 1:7
       % disp(phases(j))
        %sss
        %num2str(j-1)
        %sss
        gtLabelID(find(strcmp(phases(j), gt{2}))) = j;
        predLabelID(find(strcmp(phases(j), pred{2}))) = j;
        %gtLabelID(find(strcmp(num2str(j-1), gt{2}))) = j;
        %predLabelID(find(strcmp(num2str(j-1), pred{2}))) = j;
    end
    %disp(gtLabelID)
    % compute jaccard index, precision, recall, and the accuracy
    [jaccard(:,i), prec(:,i), rec(:,i), acc(i)] = Evaluate(gtLabelID, predLabelID, fps);
    
end

accPerVideo= acc;

% Compute means and stds
index = find(jaccard>100);
jaccard(index)=100;
meanJaccPerPhase = nanmean(jaccard, 2);
meanJaccPerVideo = nanmean(jaccard, 1);
meanJacc = mean(meanJaccPerPhase);
stdJacc = std(meanJaccPerPhase);
for h = 1:7
    jaccphase = jaccard(h,:);
    meanjaccphase(h) = nanmean(jaccphase);
    stdjaccphase(h) = nanstd(jaccphase);
end

index = find(prec>100);
prec(index)=100;
meanPrecPerPhase = nanmean(prec, 2);
meanPrecPerVideo = nanmean(prec, 1);
meanPrec = nanmean(meanPrecPerPhase);
stdPrec = nanstd(meanPrecPerPhase);
for h = 1:7
    precphase = prec(h,:);
    meanprecphase(h) = nanmean(precphase);
    stdprecphase(h) = nanstd(precphase);
end

index = find(rec>100);
rec(index)=100;
meanRecPerPhase = nanmean(rec, 2);
meanRecPerVideo = nanmean(rec, 1);
meanRec = mean(meanRecPerPhase);
stdRec = std(meanRecPerPhase);
for h = 1:7
    recphase = rec(h,:);
    meanrecphase(h) = nanmean(recphase);
    stdrecphase(h) = nanstd(recphase);
end


meanAcc = mean(acc);
stdAcc = std(acc);

% Display results
% fprintf('model is :%s\n', model_rootfolder);
disp('================================================');
disp([sprintf('%25s', 'Phase') '|' sprintf('%6s', 'Jacc') '|'...
    sprintf('%6s', 'Prec') '|' sprintf('%6s', 'Rec') '|']);
disp('================================================');
for iPhase = 1:length(phases)
    disp([sprintf('%25s', phases{iPhase}) '|' sprintf('%6.2f', meanJaccPerPhase(iPhase)) '|' ...
        sprintf('%6.2f', meanPrecPerPhase(iPhase)) '|' sprintf('%6.2f', meanRecPerPhase(iPhase)) '|']);
    disp('---------------------------------------------');
end
disp('================================================');

disp(['Mean jaccard: ' sprintf('%5.2f', meanJacc) ' +- ' sprintf('%5.2f', stdJacc)]);
disp(['Mean accuracy: ' sprintf('%5.2f', meanAcc) ' +- ' sprintf('%5.2f', stdAcc)]);
disp(['Mean precision: ' sprintf('%5.2f', meanPrec) ' +- ' sprintf('%5.2f', stdPrec)]);
disp(['Mean recall: ' sprintf('%5.2f', meanRec) ' +- ' sprintf('%5.2f', stdRec)]);

clear all;close all; clc; %copy from  webpage: http://dpfan.net/d3netbenchmark/

%SaliencyMap Path
SalMapPath = '../SalMap/';
%Models = {'LHM','DESM','CDB','ACSD','GP','LBE','CDCP','SE','MDSF','DF','CTMF','PDNet','PCF'};%'MMRNGNET2_GAN21015_vgg_GAN最小化PRELU_select'
Models = {'MIDD','ADF','FMCF','TANet','JLDCF','S2MA','AFNet','MMCI','CPD','R3Net','PoolNet','SGDL','MTMR','M3S-NIR','LENetmobilenetv2'};
% Models = {'CPD',};

%Deep Model
%DeepModels = {'DF','PDNet','CTMF','PCF'};%TPF-missing, MDSF-svm,

modelNum = length(Models);
groupNum = floor(modelNum/3)+1;

%Datasets
DataPath = '../Dataset/';
Datasets = {'VT800','VT1000','VT5000'};
% Datasets = {'NLPR','NJU2K','DES','STERE','SIP'};
%Datasets = {'GIT','SSD','DES'};
% Datasets = {'SIP'};

%Results
ResDir = '../Result_overall/';

method_colors = linspecer(groupNum);

% colors
% str=['r','r','r,','g','g','g','b','b','b','c','c','c','m','m','m','y','y','y','k','k','k','w','w','w'];

str=['g','g','b','b','y','y','c','c','k','k','m','m','g','g','r','r'];
% str=['r','g','b','y','c','m','k','r','g','b','y','c','m','k'];


datasetNum = length(Datasets);
for d = 1:datasetNum
    close all;
    dataset = Datasets{d};
    fprintf('Processing %d/%d: %s Dataset\n',d,datasetNum,dataset);
    
    matPath = [ResDir dataset '-mat/'];
    plotMetrics     = gather_the_results(modelNum, matPath, Models);
    
    %% plot the PR curves
    figure(1);
    hold on;
    grid on;
    axis([0 1 0 1]);
    title(dataset);
    xlabel('Recall');
    ylabel('Precision');
    
    for i = 1 : length(plotMetrics.Alg_names)
        if mod(i,2)==0
            plot(plotMetrics.Recall(:,i), plotMetrics.Pre(:,i), '--', 'Color', str(i), 'LineWidth', 2);   
        elseif mod(i,2)==1
            plot(plotMetrics.Recall(:,i), plotMetrics.Pre(:,i), 'Color', str(i), 'LineWidth', 2);
        end
        [~,max_idx] = max(plotMetrics.Fmeasure_Curve(:,i));
        h1 = plot(plotMetrics.Recall(max_idx,i), plotMetrics.Pre(max_idx,i), '.', 'Color', str(i),'markersize',20);
        set(get(get(h1,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    end
    legend(plotMetrics.Alg_names);
    set(gcf,'position',[0 600 560 420]);
    
    figPath = [ResDir 'Curve/'];
    if ~exist(figPath,'dir')
        mkdir(figPath);
    end
%     saveas(gcf, [figPath dataset '_PRCurve.fig'] );
    saveas(gcf, [figPath dataset '_PRCurve.fig'] );
    
    %% plot the F-measure curves
    figure(2);
    hold on;
    grid on;
    axis([0 255 0 1]);
    title(dataset);
    xlabel('Threshold');
    ylabel('F-measure');
    x = [255:-1:0]';
    for i = 1 : length(plotMetrics.Alg_names)
        if mod(i,2)==0
            plot(x, plotMetrics.Fmeasure_Curve(:,i), '--', 'Color', str(i), 'LineWidth', 2);
        elseif mod(i,2)==1
            plot(x, plotMetrics.Fmeasure_Curve(:,i), 'Color', str(i), 'LineWidth', 2);
        end
        
        [maxF,max_idx] = max(plotMetrics.Fmeasure_Curve(:,i));
        h2 = plot(255-max_idx, maxF, '.', 'Color', str(i),'markersize',20);
        set(get(get(h2,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    end
%     legend(plotMetrics.Alg_names);
    legend(plotMetrics.Alg_names);
    set(gcf,'position',[1160 600 560 420]);
    saveas(gcf, [figPath dataset '_FmCurve.fig']);
    saveas(gcf, [figPath dataset '_FmCurve.pdf'] );
end






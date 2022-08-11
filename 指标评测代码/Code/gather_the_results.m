
function plotMetrics = gather_the_results(modelNum, matPath, alg_params_exist)

alg_names                                   = cell(modelNum,1);
thrNum                                      = 256;
plotMetrics.Pre                             = zeros(thrNum,modelNum);
plotMetrics.Recall                          = zeros(thrNum,modelNum);
%plotMetrics.Fmeasure                        = zeros(thrNum,modelNum);
%plotMetrics.MAE                             = zeros(1,modelNum);


% gather the existing results
for i = 1 : modelNum
    alg_names{i}                            = alg_params_exist{1,i};
    Metrics                                 = load([matPath,alg_names{i},'.mat']);
    plotMetrics.Pre(:,i)                    = Metrics.column_Pr;
    plotMetrics.Recall(:,i)                 = Metrics.column_Rec;
    %plotMetrics.Fmeasure(:,i)               = Metrics.column_F;
    %plotMetrics.MAE(:,i)                    = Metrics.MAE;
end
plotMetrics.Fmeasure_Curve              = (1+0.3).*plotMetrics.Pre.*plotMetrics.Recall./...
    (0.3*plotMetrics.Pre+plotMetrics.Recall);
plotMetrics.Alg_names                   = alg_names;

end


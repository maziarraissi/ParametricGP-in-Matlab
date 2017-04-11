function main()
%% Pre-processing
clc; close all;

addpath ./Utilities
addpath ./Kernels
addpath ./export_fig
addpath ./Data

global ModelInfo

rng('default')

%% Setup
n = 5929413;
N = ceil(2*n/3);

%% Configuration
ModelInfo.N_batch = 1;
ModelInfo.M = 10;

ModelInfo.lrate_hyp  = 1e-3;
ModelInfo.lrate_logsigma_n = 1e-3;
ModelInfo.max_iter = N;
ModelInfo.monitor_likelihood = 10;

ModelInfo.jitter = eps;
ModelInfo.jitter_cov = eps;

%% Load & Generate Data
% Data on f(x)
load airline_data.mat
perm_data = randperm(size(X,1));
y_data = y(perm_data)';
X_data = X(perm_data,:);

y = y_data(1:N,1);
X = X_data(1:N,:);

% Normalize X
X_m = mean(X);
X_s = std(X);
X = Normalize(X, X_m, X_s);

% Normalize y
y_m = mean(y);
y_s = std(y);
y = Normalize(y, y_m, y_s);
%% Exact
N_star = n-N;
X_star = X_data(N+1:N+N_star,:);
y_star = y_data(N+1:N+N_star,1);

% Normalize X_star
X_star = Normalize(X_star, X_m, X_s);
% Normalize y_star
y_star = Normalize(y_star, y_m, y_s);


N_val = 1000;
X_val = X_data(N+1:N+N_val,:);
y_val = y_data(N+1:N+N_val,1);

% Normalize X_star
X_val = Normalize(X_val, X_m, X_s);
% Normalize y_star
y_val = Normalize(y_val, y_m, y_s);

%% Optimize model & Make Predictions
[NLML,logsigma_n,hyp] = train(X, y, X_val, y_val);
[mean_star,var_star] = predict(X_star);

fprintf(1,'MSE: %f\n', mean((mean_star-y_star).^2));
fprintf(1,'MSE of the mean of data: %f\n', mean((mean(y)-y_star).^2));

mean_star = Denormalize(mean_star, y_m, y_s);
y_star = Denormalize(y_star, y_m, y_s);
y = Denormalize(y, y_m, y_s);

fprintf(1,'RMSE: %f\n', sqrt(mean((mean_star-y_star).^2)));
fprintf(1,'RMSE of the mean of data: %f\n', sqrt(mean((mean(y)-y_star).^2)));

%%
figure(3)
bar(1./sqrt(exp(hyp(end,2:end))))
ylabel('ARD weights')
set(gca,'XTickLabel',{'Month','DayofMonth',...
    'DayOfWeek','PlaneAge','AirTime','Distance','ArrTime','DepTime'});
set(gca,'XTickLabelRotation',45)
axis tight
set(gca, 'FontSize', 14);
set(gcf, 'Color', 'w');

export_fig ./Figures/Airline_ARD.png -r300

%% Post-processing
rmpath ./Utilities
rmpath ./Kernels
rmpath ./export_fig
rmpath ./Data

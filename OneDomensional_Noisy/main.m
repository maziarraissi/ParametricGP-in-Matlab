function main()
%% Pre-processing
clc; close all;

addpath ./Utilities
addpath ./Kernels
addpath ./export_fig

global ModelInfo

rng('default')

%% Setup
N = 6000;
D = 1;
lb = 0.0*ones(1,D);
ub = 1.0*ones(1,D);
noise = 0.1;

%% Configuration
ModelInfo.N_batch = 1;
ModelInfo.M = 10;

ModelInfo.lrate_hyp  = 1e-3;
ModelInfo.lrate_logsigma_n  = 1e-3;
ModelInfo.max_iter = N;
ModelInfo.monitor_likelihood = 10;

ModelInfo.jitter = eps;
ModelInfo.jitter_cov = eps;

%% Generate Data
% Data on f(x)
f = @(x) x.*sin(4*pi*x);
X = bsxfun(@plus,lb,bsxfun(@times,   lhsdesign(N,D)    ,(ub-lb))); 
y = f(X) + noise*randn(length(X),1);

% Normalize X
X_m = mean(X);
X_s = std(X);
X = Normalize(X, X_m, X_s);
%% Exact
N_star = 400;
X_star = linspace(lb,ub,N_star)';
f_star = f(X_star);

% Normalize X_star
X_star = Normalize(X_star, X_m, X_s);

%% Train the model & Make Predictions
[NLML,logsigma_n,hyp] = train(X,y);
[mean_star,var_star] = predict(X_star);
fprintf(1,'Relative L2 error f: %e\n', (norm(mean_star-f_star,2)/norm(f_star,2)));


%% Plot results
% Denormalize inputs
X = Denormalize(X,X_m,X_s);
X_star = Denormalize(X_star,X_m,X_s);
Z = Denormalize(ModelInfo.Z,X_m,X_s);

plot_all(X,y,X_star,f_star,mean_star,var_star,Z,ModelInfo.m);

export_fig ./Figures/OneDimensional.png -r300

%%

fig = figure(2);
set(fig,'units','normalized','outerposition',[0 0 1 .4])
set(0,'defaulttextinterpreter','latex')


subplot(2,2,1)
plot(NLML)
set(gca, 'FontSize', 14);
xlabel('Iteration')
ylabel('$\mathcal{UB}$')
title('(A)')

subplot(2,2,2)
plot(exp(logsigma_n),'LineWidth',3)
set(gca, 'FontSize', 14);
xlabel('Iteration')
ylabel('$\sigma_\epsilon^2$')
title('(B)')

subplot(2,2,3)
plot(exp(hyp(:,1)),'LineWidth',3)
set(gca, 'FontSize', 14);
xlabel('Iteration')
ylabel('$\gamma^2$')
title('(C)')

subplot(2,2,4)
plot(exp(hyp(:,2)),'LineWidth',3)
set(gca, 'FontSize', 14);
xlabel('Iteration')
ylabel('$1/{w^2}$')
title('(D)')

set(gcf, 'Color', 'w');

export_fig ./Figures/OneDimensional_Convergence.png -r300

%% Post-processing
rmpath ./Utilities
rmpath ./Kernels
rmpath ./export_fig
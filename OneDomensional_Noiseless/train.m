function [NLML,logsigma_n,hyp] = train(X,y)

global ModelInfo

M = ModelInfo.M;
[N,D] = size(X);

init_params(X,y,M);

NLML = zeros(ModelInfo.max_iter,1);
logsigma_n = zeros(ModelInfo.max_iter,1);
hyp = zeros(ModelInfo.max_iter,D+1);
for iter = 1:ModelInfo.max_iter
    if ModelInfo.N_batch > 1
        idx = randperm(N, ModelInfo.N_batch);
        X_iter = X(idx,:); y_iter = y(idx);
    else
        idx = mod(iter-1,N)+1;
        X_iter = X(idx,:); y_iter = y(idx);
    end
    
    Update_K_u_inv();
        
    [NLML(iter), D_hyp, D_logsigma_n] = likelihood_UB(X_iter,y_iter);
    
    [ModelInfo.logsigma_n, ModelInfo.mt_logsigma_n, ModelInfo.vt_logsigma_n] = ...
        stochastic_update_Adam(ModelInfo.logsigma_n,...
        D_logsigma_n, ModelInfo.mt_logsigma_n, ModelInfo.vt_logsigma_n, ModelInfo.lrate_logsigma_n, iter);
    
    [ModelInfo.hyp, ModelInfo.mt_hyp, ModelInfo.vt_hyp] = ...
        stochastic_update_Adam(ModelInfo.hyp,...
        D_hyp, ModelInfo.mt_hyp, ModelInfo.vt_hyp, ModelInfo.lrate_hyp, iter);
    
    logsigma_n(iter) = ModelInfo.logsigma_n;
    hyp(iter,:) = ModelInfo.hyp;
    
    Update_K_u_inv();

    [ModelInfo.m, ModelInfo.S] = update_m_S(X_iter,y_iter);
    
    if mod(iter,ModelInfo.monitor_likelihood) == 0
        fprintf('Iteration\t%d:\t%.4f\n',iter,NLML(iter));
    end
end

end


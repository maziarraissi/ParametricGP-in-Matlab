function [mean_star,var_star] = predict(X_star)

global ModelInfo

N_star = size(X_star,1);

nn = 10000;
kk = floor(N_star/nn);
mm = mod(N_star,nn);

mean_star = zeros(N_star,1);
var_star = zeros(N_star,1);

for i = 1:kk
    fprintf('Predicting Iteration: %d out of %d\n', i, kk);
    idx = (i-1)*nn+1:i*nn;
    mean_star(idx,1) = mu(X_star(idx,:), 0);
    v_star = cov(X_star(idx,:), X_star(idx,:), 0);
    var_star(idx,1) = abs(diag(v_star)) + exp(ModelInfo.logsigma_n);
end
idx = kk*nn+1:kk*nn+mm;
mean_star(idx,1) = mu(X_star(idx,:), 0);
v_star = cov(X_star(idx,:), X_star(idx,:), 0);
var_star(idx,1) = abs(diag(v_star)) + exp(ModelInfo.logsigma_n);

end


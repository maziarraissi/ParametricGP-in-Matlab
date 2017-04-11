function init_params(X,y,M)
global ModelInfo

N = size(X,1);
idx = randperm(N, 10000);
D = size(X,2);
[~,Z] = kmeans(X(idx,:),M);

hyp = log(ones(D+1,1));
logsigma_n = -4;

m =  zeros(M,1);
S = k_u(Z, Z, hyp, 0);

ModelInfo.hyp = hyp;
ModelInfo.logsigma_n = logsigma_n;
ModelInfo.Z = Z;
ModelInfo.m = m;
ModelInfo.S = S;

ModelInfo.mt_hyp = zeros(size(hyp));
ModelInfo.vt_hyp = zeros(size(hyp));

ModelInfo.mt_logsigma_n = 0;
ModelInfo.vt_logsigma_n = 0;

end


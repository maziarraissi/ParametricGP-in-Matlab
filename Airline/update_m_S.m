function [m, S] = update_m_S(X,y)

global ModelInfo

logsigma_n = ModelInfo.logsigma_n;
sigma_n = exp(logsigma_n);

N = size(X,1);
jitter = ModelInfo.jitter;

%%% begin mu(X, 0) %%%
Z = ModelInfo.Z;
m = ModelInfo.m;
hyp = ModelInfo.hyp;

K_u_inv = ModelInfo.K_u_inv;

psi = k_u(Z, X, hyp, 0);

K_u_inv_m = K_u_inv*m;

mu = psi'*K_u_inv_m;
%%%% end mu(X, 0) %%%%


%%% begin cov(X, X, 0) %%%
S = ModelInfo.S;

K_u = k_u(Z, Z, hyp, 0);

Alpha = K_u_inv*psi;

cov_XX = k_u(X, X, hyp, 0) - Alpha'*(K_u-S)*Alpha;

%%%% end cov(X, X, 0) %%%%

K = cov_XX;

% Cholesky factorisation
[L,p]=chol(K + eye(N).*sigma_n + eye(N).*jitter,'lower');

if p > 0
    fprintf(1,'K is ill-conditioned!!\n');
end


%%% begin cov(Z, X, 0) %%%
Alpha_1 = K_u_inv*K_u;
Alpha_2 = Alpha;

cov_ZX = psi - Alpha_1'*(K_u-S)*Alpha_2;
%%%% end cov(X, X, 0) %%%%

% calculate prediction
m = m + cov_ZX*(L'\(L\(y - mu)));

alpha = L'\(L\cov_ZX');

S = S - cov_ZX*alpha;

end

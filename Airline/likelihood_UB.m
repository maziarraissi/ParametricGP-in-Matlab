function [NLML, D_NLML_hyp, D_NLML_logsigma_n]=likelihood_UB(X,y)

global ModelInfo

logsigma_n = ModelInfo.logsigma_n;
sigma_n = exp(logsigma_n);

N = size(y,1);

%%% begin mu(X, 0) %%%
Z = ModelInfo.Z;
m = ModelInfo.m;
hyp = ModelInfo.hyp;

K_u_inv = ModelInfo.K_u_inv;

psi = k_u(Z, X, hyp, 0);

K_u_inv_m = K_u_inv*m;

mu = psi'*K_u_inv_m;
%%%% end mu(X, 0) %%%%

Beta = y - mu;
NLML_1 = (Beta'*Beta)/(2*sigma_n*N);

%%% begin cov(X, X, 0) %%%
S = ModelInfo.S;

K_u = k_u(Z, Z, hyp, 0);

Alpha = K_u_inv*psi;
K_u_minus_S_times_Alpha  = (K_u-S)*Alpha;

cov = k_u(X, X, hyp, 0) - Alpha'*K_u_minus_S_times_Alpha;

%%%% end cov(X, X, 0) %%%%


NLML_2 = trace(cov)/(2*sigma_n*N);
NLML_3 = logsigma_n/2 + log(2*pi)/2;
NLML = NLML_1 + NLML_2 + NLML_3;

n_hyp = length(ModelInfo.hyp);
D_NLML_hyp = zeros(n_hyp,1);
for i=1:n_hyp
    %%% begin mu(X, i) %%%
    D_psi = k_u(Z, X, hyp, i);
    D_mu_1 = D_psi'*K_u_inv_m;
    
    D_K_u = k_u(Z, Z, hyp, i);
    D_mu_2 = -Alpha'*D_K_u*K_u_inv_m;
    
    D_mu = D_mu_1 + D_mu_2;
    %%%% end mu(X, i) %%%%
    
    D_Beta = -D_mu;
    
    D_NLML_1 = Beta'*D_Beta/(sigma_n*N);
    
    %%% begin cov(X, X, i) %%%
    D_cov_1 = k_u(X, X, hyp, i);
    
    D_Alpha = K_u_inv*D_psi;
    D_cov_2 = D_Alpha'*K_u_minus_S_times_Alpha;
    
    D_Gamma = K_u_inv*D_K_u;
    D_cov_3 = -Alpha'*D_Gamma*K_u_minus_S_times_Alpha;
    
    D_cov_4 = Alpha'*D_K_u*Alpha;
    
    D_cov_5 = D_cov_3';%-Alpha'*(K_u-S)*D_Gamma*Alpha;
    
    D_cov_6 = D_cov_2';%Alpha'*(K_u-S)*D_Alpha;
    
    D_cov = D_cov_1 - (D_cov_2 + D_cov_3 + D_cov_4 + D_cov_5 + D_cov_6);
    %%%% end cov(X, X, i) %%%%    
    
    D_NLML_2 = trace(D_cov)/(2*sigma_n*N);
    D_NLML_hyp(i) = D_NLML_1 + D_NLML_2;
end

D_NLML_logsigma_n = - NLML_1 - NLML_2 + 1/2;

end
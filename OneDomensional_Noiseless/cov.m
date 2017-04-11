function cov = cov(X_star, Xp_star, i)

global ModelInfo

Z = ModelInfo.Z;
S = ModelInfo.S;
hyp = ModelInfo.hyp;

K_u = k_u(Z, Z, hyp, 0);

K_u_inv = ModelInfo.K_u_inv;

psi_1 = k_u(Z, X_star, hyp, 0);
psi_2 = k_u(Z, Xp_star, hyp, 0);
Alpha_1 = K_u_inv*psi_1;
Alpha_2 = K_u_inv*psi_2;

if i == 0
    
    cov = k_u(X_star, Xp_star, hyp, 0) - Alpha_1'*(K_u-S)*Alpha_2;
    
else % The following is not used and is included for reference!
    
    cov_1 = k_u(X_star, Xp_star, hyp, i);
    
    D_psi_1 = k_u(Z, X_star, hyp, i);
    D_Alpha_1 = K_u_inv*D_psi_1;
    cov_2 = D_Alpha_1'*(K_u-S)*Alpha_2;
    
    D_psi = k_u(Z, Z, hyp, i);
    D_Alpha_1 = K_u_inv*D_psi;
    cov_3 = -Alpha_1'*D_Alpha_1*(K_u-S)*Alpha_2;
    
    D_K_u = k_u(Z, Z, hyp, i);
    cov_4 = Alpha_1'*(D_K_u)*Alpha_2;
    
    D_Alpha_2 = K_u_inv*D_psi;
    cov_5 = -Alpha_1'*(K_u-S)*D_Alpha_2*Alpha_2;
    
    D_psi_2 = k_u(Z, Xp_star, hyp, i);
    D_Alpha_2 = K_u_inv*D_psi_2;
    cov_6 = Alpha_1'*(K_u-S)*D_Alpha_2;
    
    cov = cov_1 - (cov_2 + cov_3 + cov_4 + cov_5 + cov_6);
end


end
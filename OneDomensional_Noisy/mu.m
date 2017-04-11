function mu = mu(X_star, i)

global ModelInfo

Z = ModelInfo.Z;
m = ModelInfo.m;
hyp = ModelInfo.hyp;

K_u_inv = ModelInfo.K_u_inv;
    
psi = k_u(X_star, Z, hyp, 0);

K_u_inv_m = K_u_inv*m;

if i == 0
    
    mu = psi*K_u_inv_m;
    
else % The following is not used and is included for reference!
    
    D_psi = k_u(X_star, Z, hyp, i);
    mu_1 = D_psi*K_u_inv_m;
    
    Alpha = K_u_inv*psi';
    D_psi = k_u(Z, Z, hyp, i);
    mu_2 = -Alpha'*D_psi*K_u_inv_m;
    
    mu = mu_1 + mu_2;
    
end

end

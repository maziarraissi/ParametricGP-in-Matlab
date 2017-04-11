function Update_K_u_inv()
global ModelInfo

Z = ModelInfo.Z;
hyp = ModelInfo.hyp;

M = size(Z,1);
jitter_cov = ModelInfo.jitter_cov;
K_u = k_u(Z, Z, hyp, 0);

[L,p]=chol(K_u + eye(M).*jitter_cov,'lower');
if p > 0
    fprintf(1,'K_u is ill-conditioned!!!\n');
end
K_u_inv = L'\(L\eye(M));
ModelInfo.K_u_inv = K_u_inv;

end


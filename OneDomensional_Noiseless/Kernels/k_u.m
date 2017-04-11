function K_u = k_u(X, Xp, hyp, i)

logsigma = hyp(1);
logtheta = hyp(2:end);
sqrt_theta = sqrt(exp(logtheta));

K_u = sq_dist(diag(1./sqrt_theta)*X',diag(1./sqrt_theta)*Xp');

K_u = exp(logsigma)*exp(-0.5*K_u);

if i > 1
    d = i-1;
    DK_u = 0.5*sq_dist((1/sqrt_theta(d))*X(:,d)',(1/sqrt_theta(d))*Xp(:,d)');
    K_u = K_u.*DK_u;
end

end

function C = sq_dist(a, b)

  C = bsxfun(@plus,sum(a.*a,1)',bsxfun(@minus,sum(b.*b,1),2*a'*b));
 
  C = max(C,0);          % numerical noise can cause C to negative i.e. C > -1e-14
end


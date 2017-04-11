function [w,mt,vt] = stochastic_update_Adam(w,grad_w,mt,vt,lrate,iter)

beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-8;

mt = mt*beta1 + (1-beta1)*grad_w;
vt = vt*beta2 + (1-beta2)*grad_w.^2;

mt_hat = mt/(1-beta1^iter);
vt_hat = vt/(1-beta2^iter);

scal = 1.0./(sqrt(vt_hat) + epsilon);

w = w - lrate.*mt_hat.*scal;


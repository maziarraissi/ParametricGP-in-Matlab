function [w,delta_w] = stochastic_update(w,delta_w,grad_w,momentum,lrate)
delta_w = momentum * delta_w + lrate * grad_w;
w = w - delta_w;


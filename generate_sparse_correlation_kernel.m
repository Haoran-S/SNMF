function A = generate_sparse_correlation_kernel(X, spars, std_noise)
% % % % % % % % % % % % % % % % % % % % % % % % % % % 
% Generate sparse correlation kernel
% Spars denote the sparsity ratio
% 
% References:
% [1] Qingjiang Shi, Haoran Sun, Songtao Lu, Mingyi Hong, and Meisam Razaviyayn. 
% "Inexact Block Coordinate Descent Methods For Symmetric Nonnegative Matrix Factorization." 
% arXiv preprint arXiv:1607.03092 (2016).
% 
% version 1.0 -- April/2016
% Written by Haoran Sun (hrsun AT iastate.edu)
% % % % % % % % % % % % % % % % % % % % % % % % % % % 

[n, r] = size(X);%each row is a data point
if spars < 1
    q = ceil(n*r*(1-spars));% the number of zeros
    y = sort(X(:), 'ascend');
    threshold = y(q);
    X(X<=threshold) = 0;
end

A = X*X';
if std_noise>0
    Noise = std_noise*randn(n, n);
    Noise = (Noise+Noise')/2;
    A = A+Noise;
end
end
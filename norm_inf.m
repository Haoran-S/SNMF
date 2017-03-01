function y = norm_inf(X)
% % % % % % % % % % % % % % % % % % % % % % % % % % % 
% Generate inf norm
% 
% References:
% [1] Qingjiang Shi, Haoran Sun, Songtao Lu, Mingyi Hong, and Meisam Razaviyayn. 
% "Inexact Block Coordinate Descent Methods For Symmetric Nonnegative Matrix Factorization." 
% arXiv preprint arXiv:1607.03092 (2016).
% 
% version 1.0 -- April/2016
% Written by Haoran Sun (hrsun AT iastate.edu)
% % % % % % % % % % % % % % % % % % % % % % % % % % % 

n = size(X, 2);
v = [];
for i = 1:n
    v = [v norm(X(:,i), Inf)];
end
y = max(v);
return
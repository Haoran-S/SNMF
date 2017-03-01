function [output, W, grad_vec, time_vec]= uniqsymnmf(A,W,K,Nnum, maxtime)
% % % % % % % % % % % % % % % % % % % % % % % % % % % 
% Implementation of sEVD algorithm
% 
% References:
% [1] Huang, Kejun, Nicholas D. Sidiropoulos, and Ananthram Swami. 
% "Non-negative matrix factorization revisited: Uniqueness and algorithm for symmetric decomposition." 
% IEEE Transactions on Signal Processing 62.1 (2014): 211-224.
% 
% [2] Qingjiang Shi, Haoran Sun, Songtao Lu, Mingyi Hong, and Meisam Razaviyayn. 
% "Inexact Block Coordinate Descent Methods For Symmetric Nonnegative Matrix Factorization." 
% arXiv preprint arXiv:1607.03092 (2016).
% 
% version 1.0 -- April/2016
% Written by Haoran Sun (hrsun AT iastate.edu)
% % % % % % % % % % % % % % % % % % % % % % % % % % % 

tic
[Us, LA] = eigs(A, K);
B= Us(:,1:K)*sqrt(LA(1:K,1:K));
output=[]; 
grad_vec = [];
time_vec = [];
for i=1:Nnum
    F=W'*B;
    [U, S, V]=svd(F);
    Q=V*U';
    W=max(0,B*Q);
    if toc>maxtime
        break;
    end
    time_vec = [time_vec toc];
    grad_vec = [grad_vec norm_inf(W-max(W-(W*(W'*W)-A*W),0))];
    output = [output norm(A-W*W','fro')^2];    
end
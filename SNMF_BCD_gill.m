function [X, obj_vec, grad_vec, time_vec] = SNMF_BCD_gill(M, maxIter, X0, maxtime)
% % % % % % % % % % % % % % % % % % % % % % % % % % % 
% Implementation of BCD algorithm
% 
% References:
% [1] Vandaele, Arnaud, et al. 
% "Efficient and non-convex coordinate descent for symmetric nonnegative matrix factorization." 
% IEEE Transactions on Signal Processing 64.21 (2016): 5571-5584.
% 
% [2] Qingjiang Shi, Haoran Sun, Songtao Lu, Mingyi Hong, and Meisam Razaviyayn. 
% "Inexact Block Coordinate Descent Methods For Symmetric Nonnegative Matrix Factorization." 
% arXiv preprint arXiv:1607.03092 (2016).
% 
% version 1.0 -- April/2016
% Written by Haoran Sun (hrsun AT iastate.edu)
% % % % % % % % % % % % % % % % % % % % % % % % % % % 

[nrow, ncol] = size(X0);
X = X0;
XXt = X*X';
vtmp = zeros(nrow, 1);
for i=1:ncol
    vtmp(i) = X(:,i)'*X(:,i);
end
obj = norm(X'*X-M, 'fro')^2;
obj_vec = [];
grad_vec = [];
time_vec = [];
iter = 0;
tic
while(iter<maxIter)
    iter = iter+1;
    for i = 1:nrow
        for j = 1:ncol  
            a = 4;
            b = 12*X(i,j);
            c = 4*(vtmp(j)-M(j,j)+XXt(i,i)+ X(i,j)^2);
            d = 4*(XXt(i,:)*X(:,j)-X(i,:)*M(:,j));
            a1 = (c-b*X(i,j))/4;
            b1 = (8*X(i,j)^3-c*X(i,j)+d)/4;
            x = ploynomialroot(a1,b1);
            XXt(i,i) = XXt(i,i)+(x-X(i,j))^2;
            XXt(:,i) = XXt(:,i)+(x-X(i,j))*X(:,j);
            XXt(i,:) = XXt(i,:)+(x-X(i,j))*X(:,j)';            
            vtmp(j) = vtmp(j)+2*(x-X(i,j))*X(i,j)+(x-X(i,j))^2;            
            mind = a/4*(x-X(i,j))^4+b/3*(x-X(i,j))^3+c/2*(x-X(i,j))^2+d*(x-X(i,j));
            obj = obj+mind;
            X(i,j) = x;
        end
    end
    if toc>maxtime
        break;
    end
    time_vec = [time_vec toc];
    obj_vec = [obj_vec abs(obj)];
    grad_vec = [grad_vec norm_inf(X-max(X-X*(X'*X-M),0))];
end
toc
end
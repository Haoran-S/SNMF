function [X, obj_vec, grad_vec, time_vec] = SNMF_cyclic_vBSUM(M, maxIter, X0, maxtime)
% % % % % % % % % % % % % % % % % % % % % % % % % % % 
% Cyclic vBSUM algorithm
% 
% References:
% [1] Qingjiang Shi, Haoran Sun, Songtao Lu, Mingyi Hong, and Meisam Razaviyayn. 
% "Inexact Block Coordinate Descent Methods For Symmetric Nonnegative Matrix Factorization." 
% arXiv preprint arXiv:1607.03092 (2016).
% 
% version 1.0 -- April/2016
% Written by Haoran Sun (hrsun AT iastate.edu)
% % % % % % % % % % % % % % % % % % % % % % % % % % %

[nrow, ncol] = size(X0); %nrow denotes the number of data points while ncol denotes the number of clusters
if nrow<ncol
    error('The # of data points should be larger than the # of clusters')
end
X = X0;
XGram = X0*X0';
XtX = X'*X;
MX = M*X;
A = M-XGram;
obj = norm(A, 'fro')^2;
obj_vec = [];
grad_vec = [];
iter = 0;
time_vec = [];
tic
while(iter<maxIter)
    iter = iter+1;
    Rndint = randperm(nrow);
    for index = 1:nrow
        i = Rndint(index);
        xi = X(i, :)';       
        Pi = XtX-xi*xi';
        sA = max(eig(Pi));
        B = sA*eye(ncol)-Pi;
        vtmp = MX(i,:)' - xi*M(i,i); 
        stmp = xi'*xi;
        obj = obj - (stmp^2+2*(xi'*Pi*xi-M(i,i)*stmp)-4*vtmp'*xi);
        for inner_iter=1:10 
            b = B*xi+vtmp+M(i,i)*xi;
            xtmp = max(b,0);
            norm_xtmp = norm(xtmp);
            if norm_xtmp==0
                xi = zeros(ncol, 1);
            else
                p = sA;
                q = -norm_xtmp;
                Delta = sqrt(q^2/4 + p^3/27);
                t = -(q/2+Delta)^(1/3) + (-q/2+Delta)^(1/3);
                xi = xtmp/norm_xtmp*t;
            end
        end
        stmp = xi'*xi;
        obj = obj + (stmp^2+2*(xi'*Pi*xi-M(i,i)*stmp)-4*vtmp'*xi);        
        XtX = Pi+xi*xi';
        MX = MX-M(:,i)*(X(i,:)-xi');
        X(i,:) = xi';
    end
    if toc>maxtime
        break;
    end
    time_vec = [time_vec toc];
    obj_vec = [obj_vec obj];
    grad_vec = [grad_vec norm_inf(X-max(X-(X*(X'*X)-M*X),0))];
end
toc
end
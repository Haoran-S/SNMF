function [output W grad_vec time_vec]= uniqsymnmf(A,W,K,Nnum, maxtime)
[N ~]=size(A);
%[Us LA]=eigs(A);
tic
%[Us LA] = ordered_eig(A, 'descend');
%if min(diag(LA(1:K, 1:K)))<=0
%   error('error')
%end
[Us LA] = eigs(A, K);
B= Us(:,1:K)*sqrt(LA(1:K,1:K));
%Q=eye(K,K);
% W=zeros(N,K);
output=[];%zeros(1,Nnum);
grad_vec = [];
time_vec = [];
for i=1:Nnum
    %[m,n]=size(B*Q);
    
    F=W'*B;
    [U S V]=svd(F);
    Q=V*U';
    W=max(0,B*Q);
    if toc>maxtime
        break;
    end
    time_vec = [time_vec toc];
    grad_vec = [grad_vec norm_inf(W-max(W-(W*(W'*W)-A*W),0))];
    output = [output norm(A-W*W','fro')^2];    
end
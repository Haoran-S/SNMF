function [X obj_vec grad_vec, time_vec] = SNMF_cyclic_vBSUM(M, maxIter, X0, maxtime)
[nrow ncol] = size(X0);%nrow denotes the number of data points while ncol denotes the number of clusters
if nrow<ncol
    error('The # of data points should be larger than the # of clusters')
end
X = X0;

XGram = X0*X0';
XtX = X'*X;
MX = M*X;
A = M-XGram;
obj = norm(A, 'fro')^2;
%obj_vec = [obj];
obj_vec = [];
grad_vec = [];
iter = 0;
time_vec = [];
tic
%sA = max(eig(M));
while(iter<maxIter)
    iter = iter+1;
    Rndint = randperm(nrow);
    for index = 1:nrow
        i = Rndint(index);
        %i = index;
        xi = X(i, :)';
        
        Pi = XtX-xi*xi';
        
        sA = max(eig(Pi));
        B = sA*eye(ncol)-Pi;
        vtmp = MX(i,:)' - xi*M(i,i);%qi
        
        stmp = xi'*xi;
        obj = obj - (stmp^2+2*(xi'*Pi*xi-M(i,i)*stmp)-4*vtmp'*xi);
        
        %obj2_vec = [norm(M-X*X', 'fro')^2];
        %xi_old = xi;
        %yi = xi;
        %t1 = 1;
        for inner_iter=1:10%max_inner_Iter
            b = B*xi+vtmp+M(i,i)*xi;
            %b = B*yi+vtmp+M(i,i)*yi;
            %b = (M(i,i)+B)*xi+vtmp;
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
            %t2 = (1+sqrt(1+4*t1^2))/2;
            %yi = xi+(t1-1)/t2*(xi-xi_old);
            %t1 = t2;
            %xi_old = xi;
            
            %X(i,:) = xi';
            %obj2 = norm(M-X*X', 'fro')^2;
            %obj2_vec = [obj2_vec obj2];
        end
%         figure(5)
%         clf
%         plot(obj2_vec)

        stmp = xi'*xi;
        obj = obj + (stmp^2+2*(xi'*Pi*xi-M(i,i)*stmp)-4*vtmp'*xi);
        
        XtX = Pi+xi*xi';
        MX = MX-M(:,i)*(X(i,:)-xi');
        X(i,:) = xi';
    end
    %A = M-X*X';
    %obj1 = norm(A, 'fro')^2;
    %obj1-obj
    if toc>maxtime
        break;
    end
    time_vec = [time_vec toc];
    obj_vec = [obj_vec obj];
    grad_vec = [grad_vec norm_inf(X-max(X-(X*(X'*X)-M*X),0))];
    %     diff = abs(obj_vec(end-1)-obj)/obj_vec(end-1)
    %     if diff <= 1e-3
    %        break;
    %     end
end
% figure(3)
% clf
% plot(obj_vec)
%X = X*sqrt(maxM);
toc
end
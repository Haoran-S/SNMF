function [X obj_vec, grad_vec, time_vec] = SNMF_cyclic_BSUM(M, maxIter, X0, maxtime)
[nrow ncol] = size(X0);
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

            p = (3*a*c-b^2)/3/a^2;
            q = (9*a*b*c-27*a^2*d-2*b^3)/27/a^3;
            
            if c>b^2/3/a
                DEL = sqrt(q^2/4+1/27*p^3);
                x = (DEL+q/2)^(1/3)-(DEL-q/2)^(1/3);%nthroot(q/2-DEL, 3)+nthroot(q/2+DEL, 3);
            else
                %[b c]
                %c-b^2/3/a
                stmp = b^3/27/a^3-d/a;
                x = sign(stmp)*abs(stmp)^(1/3);%nthroot(stmp, 3);
            end
            
            x = max(x, 0);
            
            XXt(i,i) = XXt(i,i)+(x-X(i,j))^2;
            XXt(:,i) = XXt(:,i)+(x-X(i,j))*X(:,j);
            XXt(i,:) = XXt(i,:)+(x-X(i,j))*X(:,j)';
            
            vtmp(j) = vtmp(j)+2*(x-X(i,j))*X(i,j)+(x-X(i,j))^2;
            
            mind = a/4*(x-X(i,j))^4+b/3*(x-X(i,j))^3+c/2*(x-X(i,j))^2+d*(x-X(i,j));
            obj = obj+mind;
            X(i,j) = x;
        end
        %obj_vec = [obj_vec obj];
        %     diff = abs(obj_vec(end-1)-obj)/obj_vec(end-1)
        %     if diff <= 1e-3
        %        break;
        %     end
    end
    if toc>maxtime
        break;
    end
    time_vec = [time_vec toc];
    obj_vec = [obj_vec abs(obj)];
    %obj_vec = [obj_vec obj];
    grad_vec = [grad_vec norm_inf(X-max(X-((X*X')*X-X*M),0))];
end
toc
end
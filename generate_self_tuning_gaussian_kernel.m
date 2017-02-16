function A = generate_self_tuning_gaussian_kernel(X)
[n r] = size(X);%each row is a data point
D = dist2(X, X);
q = floor(log2(n))+1;
sigma = zeros(n,1);
threshold = sigma;
for i=1:n
    y = D(i,:);
    y(i) = [];
    y = sort(y, 'ascend');
    sigma(i) = sqrt(y(2));%the 7-th neighbor
    threshold(i) = y(q);
end

E = zeros(n,n);
for i=1:n
    for j=1:n
        if i~=j
            E(i,j) = exp(-D(i,j)/sigma(i)/sigma(j));
        end
    end
end

%sparsify the network
for i=1:n
    E(i, D(i,:)>threshold(i)) = 0; 
end
for i=1:n
    for j=i:n
        stmp = max(E(i,j), E(j,i));%make E symmetric
        E(i,j) = stmp;
        E(j,i) = stmp;
    end
end%end of sparsification

d = sum(E, 2);
clear D;
for i=1:n
    for j=1:n
        A(i,j) = E(i,j)/sqrt(d(i))/sqrt(d(j));
    end
end
end
function A = generate_sparse_correlation_kernel(X, spars, std_noise)%spars denote the sparsity ratio
[n r] = size(X);%each row is a data point
if spars < 1
    q = ceil(n*r*(1-spars));% the number of zeros
    [y ind] = sort(X(:), 'ascend');
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
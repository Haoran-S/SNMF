function y = norm_inf(X)
n = size(X, 2);
v = [];
for i = 1:n
    v = [v norm(X(:,i), Inf)];
end
y = max(v);
return
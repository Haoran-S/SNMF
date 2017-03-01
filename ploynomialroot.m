function xopt = ploynomialroot(a,b)
delta = 4*a^3+27*b^2;
if delta <=0
    r = 2*sqrt(-a/3);
    theta = atan2(sqrt(-delta/108),-b/2)/3;
    ymax=0; xopt =0;
    for k = 0:2:4
        x = r*cos(theta+((k*pi)/3));
        if x >=0
            y = x^4/4+a*x^2/2+b*x;
            if y < ymax
                ymax=y;
                xopt=x;
            end
        end
    end
else
    z = sqrt(delta/27);
    x = cubicRoot(1/2*(-b+z))+cubicRoot(1/2*(-b-z));
    y = x^4/4+a*x^2+b*x;
    if y<0 && x>=0
        xopt = x;
    else
        xopt=0;
    end
end
function y =cubicRoot(x)
y = sign(x) * abs(x)^(1/3);

% The lorenz equations
function dsdt = lorenz(t,s,sigma,b,r,T,f)
dsdt = zeros(3,1);
dsdt(1) = -sigma*s(1) + sigma*s(2) + f(1,floor(t/T)+1);
dsdt(2) = -s(1)*s(3) + r*s(1) - s(2) + f(2,floor(t/T)+1);
dsdt(3) = s(1)*s(2) - b*s(3) + f(3,floor(t/T)+1);
end
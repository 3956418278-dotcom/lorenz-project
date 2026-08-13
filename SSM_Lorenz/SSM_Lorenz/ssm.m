% Compute the output of a linear SSM given its input
function [Y_tier,y_tier] = ssm(sys,f,x0)

Fs = 16;
tg = 4096;
N1 = Fs*tg;

[A,B,C] = ssm2ABC(sys);
n = length(A);
m = size(C,1);

% default x0: x0 stored in sys
if nargin < 3 || isempty(x0)
    [~,~,~,x0] = ssm2ABC(sys);
end

if isempty(B)
    B = zeros(n,m);
end

x = zeros(n,N1);
x(:,1) = x0;
y_tier = zeros(m,N1);
for j = 1:N1-1
    x(:,j+1) = A*x(:,j) + B*f(:,j);
    y_tier(:,j) = C*x(:,j);
end
y_tier(:,N1) = C*x(:,N1);

Y_tier = fft_single(y_tier);
end
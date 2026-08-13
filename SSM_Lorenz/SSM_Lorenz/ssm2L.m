% Computing the LFRF (i.e. transfer function) given a linear SSM
function L = ssm2L(sys,nu)

% Default: fre = 0 Hz, i.e., steady-state
if nargin < 2 || isempty(nu)
    nu = 0;
end

T = 1/16;

[A,B,C] = ssm2ABC(sys);
n = length(A);
w = nu * 2 * pi; % Convert to circular frequency
L = C * (exp(1i*w*T) * eye(n) - A)^(-1) * B;

end


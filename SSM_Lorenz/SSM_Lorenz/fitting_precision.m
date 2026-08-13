% Compute the fitting precision of the linear SSM
function R = fitting_precision(y,y_tier,w)
err = y - y_tier;

% Default weighting vector w
if nargin < 3 || isempty(w)
    m = size(y,1);
    w = 1/m * ones(m,1);
end

sim_err = mean(sqrt(sum((err.*w).^2))) / mean(sqrt(sum((y.*w).^2)));
R = sqrt(1-sim_err.^2);
end
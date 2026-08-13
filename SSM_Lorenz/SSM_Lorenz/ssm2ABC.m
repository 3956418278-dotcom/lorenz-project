% Get the parameters of a linear SSM
function [A,B,C,x0,K] = ssm2ABC(sys)

if class(sys) == "idss"
    A = sys.A;
    B = sys.B;
    C = sys.C;
    x0 = sys.Report.Parameters.X0(:,1);
    K = sys.K;
end

if class(sys) == "cell"
    A = sys{1};
    B = sys{2};
    C = sys{3};
    x0 = sys{4}(:,1);
    K = sys{5};
end

end


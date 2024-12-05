function [f,G]=rosenbrock(X)

f=100*(X(2)-X(1)^2)^2+(1-X(1))^2;

if nargout>1  % si le gradient est demand� alors on le calcule
    
    G=[100*(4*X(1)^3-4*X(1)*X(2))+2*X(1)-2
        100*(2*X(2)-2*X(1)^2)];
    
end

end
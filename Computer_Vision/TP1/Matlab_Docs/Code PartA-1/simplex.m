function S = simplex(X,stepsize)
% 
% Generate a simplex around the point X. This function is useful
% to initialize the Nelder&Mead optimization algorithm (nelder function)
% 
%     S = simplex(X,stepsize)
%
% Input: X = column vector, location of the origin of the simplex
%        stepsize = specification of a fixed step size for the 
%            excursions from X.
%
% Ouput: S = n by (n+1) matrix
%          = [ X(1)  X(1)+stepsize           X(1)  ...           X(1)
%              X(2)           X(1)  X(2)+stepsize  ...           X(2)
%              ...             ...            ...  ...            ...
%              X(n)           X(n)           X(n)  ...  X(n)+stepsize ];
%

% G. Laurent, Nov, 2013
%
% This code comes with no guarantee or warranty of any kind.

    S = X*ones(1,length(X))+stepsize*eye(length(X));
    S = [X S];
end 
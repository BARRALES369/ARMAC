function [grad,num] = gradest(fun,x0,stepsize)
%
% Gradient estimate of a function of n variables using forward-differencing
% 
%     grad = gradest(fun,x0,stepsize)
%
% Inputs: fun = analytical function to differentiate. fun must
%             be a function of the vector or array x0.
%         x0 = vector location at which to differentiate fun
%             If x0 is an nxm array, then fun is assumed to be
%             a function of n*m variables. 
%         stepsize = specification of a fixed step size for the excursions 
%             from x0.
%
% Output: grad = column vector of first partial derivatives of fun.
%        

% G. Laurent, Nov, 2013
%
% This code comes with no guarantee or warranty of any kind.

% sx = length(x0);
% grad = zeros(sx,1);
% for i = 1:sx
%     
%     xp=x0;
%     xp(i)=xp(i)+stepsize;
%     xm=x0;
%     xm(i)=xm(i)-stepsize;
%     grad(i)=(feval(fun,xp)-feval(fun,xm))/stepsize/2;
% end
% 
% num=2*sx;

sx = length(x0);
grad = zeros(sx,1);
f0=feval(fun,x0);
for i = 1:sx
    
    xp=x0;
    xp(i)=xp(i)+stepsize;
    grad(i)=(feval(fun,xp)-f0)/stepsize;
end

num=sx+1;

end 
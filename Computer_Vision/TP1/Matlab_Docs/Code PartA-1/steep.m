function [x,histout,costdata] = steep(x0,f,tol,maxit,budget,stepsize)
%
% Steepest descent with Armijo rule and polynomial linesearch.
%
%     [x,histout,costdata] = steep(x0,f,tol,maxit)
%
% Inputs: x0 = initial iterate
%         f = objective function,
%         tol = termination criterion norm(grad) < tol
%             optional, default = 1.d-6
%         maxit = maximum iterations (optional) default = 1000
%         budget = max f evals (optional) default=maxit*10
%             The iteration will terminate after the iteration that
%             exhausts the budget
%         stepsize = specification of a fixed step size for the gradient
%             estimation (optional)
%             If not provided the calling sequence for f should be
%             [fout,gout]=f(x) where fout is a scalar
%             and gout is a COLUMN vector (f gradient)
%             If provided the calling sequence for f should be
%             fout=f(x) where fout is a scalar
%
% Outputs: x = solution
%          histout = iteration history
%              Each row of histout is
%              [iteration count, f, norm(grad), number of step length reductions]
%          costdata = [num f, num grad, num hess] (for steep, num hess=0)
%
% See also: nelder, bfgs


% C. T. Kelley, Dec 20, 1996
%
% modified by G. Laurent, Nov, 2013
%
% This code comes with no guarantee or warranty of any kind.

%
% linesearch parms
%
bhigh=.5; blow=.1;
%
%
alp=1.d-4;
if nargin < 4
    maxit=1000;
end
if nargin < 3
    tol=1.d-6;
end
if nargin < 5
    budget=maxit*10;
end

usergrad=1;
if nargin == 6
    usergrad=0;
end
itc=1; xc=x0;
numf=0; numg=0; numh=0;
if usergrad==1
    [fc,gc]=feval(f,xc); numf=numf+1; numg=numg+1;
else
    fc=feval(f,xc);
    [gc,num]=gradest(f,xc,stepsize);
    numf=numf+1+num;
end
ithist=zeros(maxit,4);
ithist(1,1)=itc-1; ithist(1,2) = fc; ithist(1,3)=norm(gc); ithist(1,4)=0;
while(norm(gc) > tol && itc <= maxit && numf < budget)
    %
    %       fixup for very long steps, see (3.50) in the book
    %
    lambda=min(1,100/(1+norm(gc))); xt=xc-lambda*gc; ft=feval(f,xt);
    numf=numf+1;
    iarm=0; itc=itc+1;
    fgoal=fc-alp*lambda*(gc'*gc);
    %
    %       polynomial line search
    %
    q0=fc; qp0=-gc'*gc; lamc=lambda; qc=ft;
    while(ft > fgoal)
        iarm=iarm+1;
        if iarm==1
            lambda=polymod(q0, qp0, lamc, qc, blow, bhigh);
        else
            lambda=polymod(q0, qp0, lamc, qc, blow, bhigh, lamm, qm);
        end
        qm=qc; lamm=lamc; lamc=lambda;
        xt=xc-lambda*gc;
        ft=feval(f,xt); numf = numf+1; qc=ft;
        if(iarm > 10)
            disp('Exiting steep: Armijo error in steepest descent ')
            x=xc; histout=ithist(1:itc,:); costdata=[numf, numg, numh];
            return;
        end
        fgoal=fc-alp*lambda*(gc'*gc);
    end
    xc=xt; %[fc,gc]=feval(f,xc); numf=numf+1; numg=numg+1;
    if usergrad==1
        [fc,gc]=feval(f,xc); numf=numf+1; numg=numg+1;
    else
        fc=feval(f,xc);
        [gc,num]=gradest(f,xc,stepsize);
        numf=numf+1+num;
    end
    ithist(itc,1)=itc-1; ithist(itc,2) = fc;
    ithist(itc,3)=norm(gc); ithist(itc,4)=iarm;
end
x=xc;
histout=ithist(1:itc,:); costdata=[numf, numg, numh];

if itc > maxit
    disp(' ');
    disp('Exiting steep: Maximum number of iterations has been exceeded');
end
if norm(gc) <= tol
    disp(' ');
    disp('Exiting steep: The current x satisfies the termination criteria norm(gc) <= tol');
end
if numf > budget
    disp(' ');
    disp('Exiting steep: Maximum number of function evaluations has been exceeded');
end
disp(['  Iteration number: ' num2str(size(histout,1))]);
disp(['  Number of function evaluation: ' num2str(costdata(1))]);
disp(['  Gradient norm: ' num2str(norm(gc))]);
disp(['  Function value f: ' num2str(fc)]);
disp(['  Minimizer X: [ ' num2str(x') ' ]']);
disp(' ');

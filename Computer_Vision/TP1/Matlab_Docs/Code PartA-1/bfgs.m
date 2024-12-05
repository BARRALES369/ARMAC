function [x,histout,costdata] = bfgs(x0,f,tol,maxit,budget,stepsize)
%
% BFGS optimizer with polynomial line search and Steve Wright storage of
% H^-1.
%
%     [x,histout,costdata] = bfgswopt(x0,f,tol,maxit,hess0)
%
% Inputs: x0 = initial iterate
%         f = objective function,
%         tol = termination criterion norm(grad) < tol
%             optional, default = 1.d-6
%         maxit = maximum iterations (optional) default = 20
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
%          costdata = [num f, num grad, num hess]
%              (num hess=0 always, in here for compatibility with steep.m)
%
% See also: nelder, steep

% C. T. Kelley, July 17, 1997
%
% modified by G. Laurent, Nov, 2013
%
% This code comes with no guarantee or warranty of any kind.

% At this stage all iteration parameters are hardwired in the code.
%
blow=.1; bhigh=.5;
numf=0; numg=0; numh=0;
if nargin < 4
    maxit=20;
end
if nargin < 3
    tol=1.d-6;
end
itc=1; xc=x0;
maxarm=10; nsmax=50; debug=0;
%
n=length(x0);
userhess=0;
% if nargin == 5
% userhess=1;
% end
if nargin < 5
    budget=maxit*10;
end
usergrad=1;
if nargin == 6
    usergrad=0;
end
if usergrad==1
    [fc,gc]=feval(f,xc); numf=numf+1; numg=numg+1;
else
    fc=feval(f,xc);
    [gc,num]=gradest(f,xc,stepsize);
    numf=numf+1+num;
end
ithist=zeros(maxit,4);
ithist(1,1)=itc-1; ithist(1,2) = fc; ithist(1,3)=norm(gc); ithist(1,4)=0;
if debug==1
    ithist(itc,:)
end
go=zeros(n,1);
alpha=zeros(nsmax,1); beta=alpha;
sstore=zeros(n,nsmax); ns=0;
%
%	dsdp = - H_c^{-1} grad_+ if ns > 0
%
while(norm(gc) > tol && itc <= maxit && numf < budget)
    dsd=-gc;
    dsdp=-gc;
    if userhess==1 dsdp=feval(hess0,dsd); end
    if (ns>1)
        if userhess==1
            dsdp=bfgsw(sstore,alpha,beta,ns,dsd,hess0);
        else
            dsdp=bfgsw(sstore,alpha,beta,ns,dsd);
        end
    end
    %
    %
    % compute the direction
    %
    if (ns==0)
        dsd=-gc;
        if userhess==1 dsd=feval(hess0,dsd); end
    else
        xi=-dsdp;
        b0=-1/(y'*s);
        zeta=(1-1/lambda)*s+xi;
        a1=b0*b0*(zeta'*y);
        a1=-b0*(1 - 1/lambda)+b0*b0*y'*xi;
        a=-(a1*s+b0*xi)'*gc;
        %
        %		We save go=s'*g_old just so we can use it here
        %		and avoid some cancellation error
        %
        alphatmp=a1+2*a/go;
        b=-b0*go;
        %
        %
        dsd=a*s+b*xi;
    end
    %
    %
    %
    if (dsd'*gc > -1.d-6*norm(dsd)*norm(gc))
        disp('Warning: Loss of descent')
        %[itc, dsd'*gc]
        dsd=-gc;
        ns=0;
    end
    lambda=1;
    %
    %       fixup against insanely long steps see (3.50) in the book
    %
    lambda=min(1,100/(1 + norm(gc)));
    xt=xc+lambda*dsd; ft=feval(f,xt); numf=numf+1;
    itc=itc+1;
    old=1;
    if old==0
        goal=fc+1.d-4*(gc'*dsd); iarm=0;
        if ft > goal
            [xt,iarm,lambda]=polyline(xc,fc,gc,dsd,ft,f,maxarm);
            if iarm==-1
                x=xc; histout=ithist(1:itc,:); costdata=[numf, numg, numh];
                disp(' ');
                disp('Exiting bfgs: Line search failure');
                return;
            end
        end
    end
    if old==1
        iarm=0; goalval=.0001*(dsd'*gc);
        q0=fc; qp0=gc'*dsd; lamc=lambda; qc=ft;
        while(ft > fc + lambda*goalval )
            iarm=iarm+1;
            if iarm==1
                lambda=polymod(q0, qp0, lamc, qc, blow, bhigh);
            else
                lambda=polymod(q0, qp0, lamc, qc, blow, bhigh, lamm, qm);
            end
            qm=qc; lamm=lamc; lamc=lambda;
            xt=xc+lambda*dsd;
            ft=feval(f,xt); qc=ft; numf=numf+1;
            if(iarm > maxarm)
                x=xc; histout=ithist(1:itc,:); costdata=[numf, numg, numh];
                disp(' ');
                disp('Exiting bfgs: Too many backtracks in BFGS line search');
                return;
            end
        end
    end
    s=xt-xc; y=gc; go=s'*gc;
    %        lambda=norm(s)/norm(dsd);
    xc=xt;
    if usergrad==1
        [fc,gc]=feval(f,xc); numf=numf+1; numg=numg+1;
    else
        fc=feval(f,xc);
        [gc,num]=gradest(f,xc,stepsize);
        numf=numf+1+num;
    end
    
    y = gc-y;
    %
    %   restart if y'*s is not positive or we're out of room
    %
    if y'*s <= 0
        disp('Warning: Loss of positivity');
        %[ns, y'*s]
        ns=0;
    elseif ns==nsmax
        disp('Warning: Out of storage');
        %[ns, y'*s]
        ns=0;
    else
        ns=ns+1; sstore(:,ns)=s;
        if(ns>1)
            alpha(ns-1)=alphatmp;
            beta(ns-1)=b0/(b*lambda);
        end
    end
    ithist(itc,1)=itc-1; ithist(itc,2) = fc;
    ithist(itc,3)=norm(gc); ithist(itc,4)=iarm;
    if debug==1
        ithist(itc,:)
    end
    
end
x=xc; histout=ithist(1:itc,:); costdata=[numf, numg, numh];

if itc > maxit
    disp(' ');
    disp('Exiting bfgs: Maximum number of iterations has been exceeded');
end
if norm(gc) <= tol
    disp(' ');
    disp('Exiting bfgs: The current x satisfies the termination criteria norm(gc) <= tol');
end
if numf > budget
    disp(' ');
    disp('Exiting bfgs: Maximum number of function evaluations has been exceeded');
end
disp(['  Iteration number: ' num2str(size(histout,1))]);
disp(['  Number of function evaluation: ' num2str(costdata(1))]);
if usergrad==1
    disp(['  Number of gradient evaluation: ' num2str(costdata(2))]);
end
disp(['  Gradient norm: ' num2str(norm(gc))]);
disp(['  Function value f: ' num2str(fc)]);
disp(['  Minimizer X: [ ' num2str(x') ' ]']);
disp(' ');



%
% bfgsw
%
% C. T. Kelley, Dec 20, 1996
%
% This code comes with no guarantee or warranty of any kind.
%
% This code is used in bfgs.m
%
% There is no reason to ever call this directly.
%
% form the product of the bfgs approximate inverse Hessian
% with a vector using the Steve Wright method
%
function dnewt=bfgsw(sstore,alpha,beta,ns,dsd,hess0)
userhess=0; if nargin==6 userhess=1; end
dnewt=dsd;
if userhess==1 dnewt=feval(hess0,dnewt); end
if (ns<=1) return; end;
dnewt=dsd; n=length(dsd);
if userhess==1 dnewt=feval(hess0,dnewt); end
sigma=sstore(:,1:ns-1)'*dsd; gamma1=alpha(1:ns-1).*sigma;
gamma2=beta(1:ns-1).*sigma;
gamma3=gamma1+beta(1:ns-1).*(sstore(:,2:ns)'*dsd);
delta=gamma2(1:ns-2)+gamma3(2:ns-1);
dnewt=dnewt+gamma3(1)*sstore(:,1)+gamma2(ns-1)*sstore(:,ns);
if(ns <=2) return; end
dnewt=dnewt+sstore(1:n,2:ns-1)*delta(1:ns-2);
%
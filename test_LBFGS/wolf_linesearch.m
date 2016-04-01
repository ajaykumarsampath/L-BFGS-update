function [ alpha ] = wolf_linesearch( fun_quad,Grad,x,d,ops)
%
% This function is the line search algorithm statisfying strong
% Wolfe conditions. Algorithm 3.5 on pages 60-61 in Nocedal and
% Wright. In particular for QP.
%
% Syntax : [ alpha ] = wolf_linesearch( fun_quad,Grad,x,d)
%
% INPUT :   fun_quad  :  QP with quadratic term and a linear term
%               Grad  :  The gradient at x0
%                 x0  :  x0
%                  d  :  Direction
%
% OUTPUT :     alpha  :  step size
%

if(isfield(ops,'alpha'))
    alpha0=ops.alpha;
else
    alpha0=0;
end

Q=fun_quad.Q;
b=fun_quad.b;

% phi_c the value function at the current direction
% alpha_p previous alpha
% alpah_c current alpha
% phi_p  value function at the previous direction

alpha_max=1;
% parameter for sufficient decrease condition
c1 = 1e-4;
% parameter for curvature condition
c2 = 0.9;
ops_zoom.c1=c1;
ops_zoom.c2=c2;
ops_zoom.iter_max=ops.iter_max;

phi0=0.5*x'*Q*x+b'*x;
alpha=0;
alpha_p=alpha0;
alpha_c=rand(1)*alpha_max;
%alpha_c=1;
phi_p=phi0;
%Grad_p=Grad;
i=1;

 %{
    if(alpha_p==alpha_c)
        alpha_p=alpha0;
        alpha_c=rand(1)*alpha_max;
    end
    %}
%while(i<ops.iter_max)
while(abs(Grad'*d)>1e-8)
    %
    if(alpha_p==alpha_c)
        alpha_p=alpha0;
        alpha_c=rand(1)*alpha_max;
        phi_p=phi0;
    end
    %}
    % New direction
    Xkk=x+alpha_c*d;
    % value function at Xkk
    phi_c=0.5*Xkk'*Q*Xkk+b'*Xkk;
    if (phi_c > phi0+c1*alpha_c*Grad'*d) || (phi_c >= phi_p && i>1)
        alpha_k=zoom_sectioning(fun_quad,Grad,x,d,alpha_p,alpha_c,ops_zoom);
        alpha=alpha_k;
        %i=ops.iter_max+10;
        %break;
        return
    end
    % gradient at Xkk
    Grad_c=(Q*Xkk+b)'*d;
    
    if abs(Grad_c')<=-c2*Grad'*d
        %if c2*Grad'*d<=Grad_c
        %i-2
        alpha=alpha_c;
        %break;
        return
    end
    %end
    
    if Grad_c>=0
        alpha_k=zoom_sectioning(fun_quad,Grad,x,d,alpha_c,alpha_p,ops_zoom);
        alpha=alpha_k;
        %break;
        return
        %i=ops.iter_max+10;
    end
    
    %
    alpha_p=alpha_c;
    phi_p=phi_c;
    alpha_c=alpha_p+(alpha_max-alpha_p)*rand(1);
    %{
     alpha_c=alpha_p+(alpha_max-alpha_p)*rand(1);
       %alpha_p=alpha_c;
       %phi_p=phi_c;
    %}
    i=i+1;
    if(i>ops.iter_max)
        disp('max iterations line search')
        alpha=alpha_c;
        break
    end
    %}
end
end






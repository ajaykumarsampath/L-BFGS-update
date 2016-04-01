function [ alpha ] = wolf_linesearch( obj,Grad,Z,Y,d,ops)
%
% This function is the line search algorithm statisfying strong
% Wolfe conditions. Algorithm 3.5 on pages 60-61 in Nocedal and
% Wright. In particular for QP.
%
% Syntax : [ alpha ] = wolf_linesearch( fun_quad,Grad,x,d)
%
% INPUT :       obj   :  Object of the algorithm
%               Grad  :  Gradient of the FBE
%               Y     :
%               d     :  Direction of the FBE
%
% OUTPUT :     alpha  :  step size
%

if(isfield(ops,'alpha'))
    alpha0=ops.alpha;
else
    alpha0=0;
end


tree=obj.SysMat_.tree;
V=obj.SysMat_.V;
Nd=lenght(tree.stage);
Ns=lenght(tree.leaves);
non_leaf=Nd-Ns;
x0=ops.x0;
phi0=0;

separ_vars=ops.separ_vars;
lambda=obj.algo_details.ops_FBS.alpha;

curv_dir=0;
for i=1:non_leaf
    phi0=phi0+Z.X(:,i)'*V.Q*Z.X(:,i)+Z.U(:,i)'*V.R*Z.U(:,i)+...
        0.5*lambda*(separ_vars.y'*separ_vars.y)...
        +Y.y(:,i)'*separ_vars.y(:,i);
    curv_dir=curv_dir+Grad.y(:,i)'*d.y(:,i);
end

for i=1:Ns
    phi0=phi0+Z.X(:,non_leaf+i)'*V.Vf*Z.X(:,non_leaf+i)+...
        Y.yt{i}'*separ_vars.yt{i}+0.5*lambda*norm(separ_vars.yt{i})^2;
    curv_dir=curv_dir+Grad.yt{i}'*d.yt{i};
end
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
ops_zoom.phi0=phi0;
ops_zoom.curv_dir=curv_dir;
ops_zoom.x0=x0;

alpha=0;
alpha_p=alpha0;
alpha_c=rand(1)*alpha_max;
phi_p=phi0;

i=1;

while(abs(curv_dir)>1e-8)
    %
    if(alpha_p==alpha_c)
        alpha_p=alpha0;
        alpha_c=rand(1)*alpha_max;
        phi_p=phi0;
    end
    %}
    % New direction
    Ynew.y=Y.y+alpha_c*d.y;
    Ynew.yt=Y.yt+alpha_c*d.yt;
    
    % value function at Xkk
    [Grad_new,Znew,details_new] =obj.grad_dual_envelop(Ynew,x0);
    separ_var_new.y=details_new.Hx-details_new.T.y;
    separ_var_new.yt=details_new.Hx_term-details_new.T.yt;
    
    phi_c=0;
    new_curv_dir=0;
    for i=1:non_leaf
        phi_c=phi_c+Znew.X(:,i)'*V.Q*Znew.X(:,i)+Znew.U(:,i)'...
            *V.R*Znew.U(:,i)+0.5*lambda*(separ_var_new.y'*...
            separ_var_new.y)+Ynew.y(:,i)'*separ_var_new.y(:,i);
        new_curv_dir=new_curv_dir+Grad_new.y(:,i)'*d.y(:,i);
    end
    
    for i=1:Ns
        phi_c=phi_c+Znew.X(:,non_leaf+i)'*V.Vf*Znew.X(:,non_leaf+i)+...
            Ynew.yt{i}'*separ_var_new.yt{i}+0.5*lambda*...
            norm(separ_var_new.yt{i})^2;
        new_curv_dir=new_curv_dir+Grad_new.yt{i}'*d.yt{i};
    end
    
    
    if (phi_c > phi0+c1*alpha_c*curv_dir) || (phi_c >= phi_p && i>1)
        alpha_k=obj.zoom_sectioning(Y,d,alpha_p,alpha_c,ops_zoom);
        alpha=alpha_k;
        %i=ops.iter_max+10;
        %break;
        return
    end
    % gradient at Xkk
    
    if abs(new_curv_dir)<=-c2*curv_dir
        %if c2*Grad'*d<=Grad_c
        %i-2
        alpha=alpha_c;
        %break;
        return
    end
    %end
    
    if Grad_c>=0
        alpha_k=obj.zoom_sectioning(Y,d,alpha_c,alpha_p,ops_zoom);
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






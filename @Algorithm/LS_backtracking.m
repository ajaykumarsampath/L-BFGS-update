function [ alpha,details_LS ] = LS_backtracking(obj,Grad,Z,Y,d,ops)
%
% This function is the line search algorithm to calculate the 
% step-size using the Goldstein_conditinons given in the equation 3.11 
% in Wright and Nocedel.
%
% Syntax : [ alpha ] = Goldstein_conditions( fun_quad,Grad,x,d)
%
% INPUT :       obj   :  Object of the algorithm
%               Grad  :  Gradient of the FBE
%               Y     :
%               d     :  Direction of the FBE
%
% OUTPUT :     alpha  :  step size
%


tree=obj.SysMat_.tree;
%sys=obj.SysMat_.sys;
V=obj.SysMat_.V;
Nd=length(tree.stage);
Ns=length(tree.leaves);
non_leaf=Nd-Ns;

separ_vars=ops.separ_vars;
lambda=obj.algo_details.ops_FBE.lambda;

phi0=0;
curv_dir=0;
for j=1:non_leaf
    phi0=phi0+tree.prob(j)*Z.X(:,j)'*V.Q*Z.X(:,j)+tree.prob(j)*Z.U(:,j)'*V.R*Z.U(:,j)+...
        0.5*lambda*norm(separ_vars.y(:,j))^2+Y.y(:,j)'*separ_vars.y(:,j);
    curv_dir=curv_dir+Grad.y(:,j)'*d.y(:,j);
end

for j=1:Ns
    phi0=phi0+tree.prob(non_leaf+j)*Z.X(:,non_leaf+j)'*V.Vf{j}*Z.X(:,non_leaf+j)+...
        Y.yt{j}'*separ_vars.yt{j}+0.5*lambda*norm(separ_vars.yt{j})^2;
    curv_dir=curv_dir+Grad.yt{j}'*d.yt{j};
end
phi0=-phi0;

% phi0,phi_c are the value function at the current and previous direction
% phi_diff=phi_c-phi0

alpha_c=1;
beta=0.5;
i=1;
details_LS.term_LS=0;
details_LS.inner_loops=0;
x0=ops.x0;
% difference in the objective functions  

while(abs(curv_dir)>1e-8)
    %% New direction--dual variable
    Ynew.y=Y.y+alpha_c*d.y;
    for j=1:Ns
        Ynew.yt{j,1}=Y.yt{j,1}+alpha_c*d.yt{j,1};
    end
    
    %
    [Grad_new,Znew,details_new] =obj.grad_dual_envelop(Ynew,x0);
    separ_var_new.y=details_new.Hx-details_new.T.y;
    
    for j=1:Ns
        separ_var_new.yt{j,1}=details_new.Hx_term{j,1}-details_new.T.yt{j,1};
    end 
    
    phi_c=0;
    new_curv_dir=0;
    for j=1:non_leaf
        phi_c=phi_c+tree.prob(j)*Znew.X(:,j)'*V.Q*Znew.X(:,j)+tree.prob(j)*Znew.U(:,j)'...
            *V.R*Znew.U(:,j)+0.5*details_new.lambda*(separ_var_new.y(:,j)'*...
            separ_var_new.y(:,j))+Ynew.y(:,j)'*separ_var_new.y(:,j);
        new_curv_dir=new_curv_dir+Grad_new.y(:,j)'*d.y(:,j);
    end
    
    for j=1:Ns
        phi_c=phi_c+tree.prob(non_leaf+j)*Znew.X(:,non_leaf+j)'*V.Vf{j}*Znew.X(:,non_leaf+j)+...
            Ynew.yt{j}'*separ_var_new.yt{j}+0.5*details_new.lambda*norm(separ_var_new.yt{j})^2;
    end
    phi_c=-phi_c;
    
    phi_diff=phi_c-phi0;
    if(phi_diff>0)
        alpha_c=beta*alpha_c;
        i=i+1;
    else
        details_LS.inner_loops=i;
        alpha=alpha_c;
        break;
    end 
    if(i>ops.iter_max)
        details_LS.term_LS=1;
        alpha=0;
        details_LS.inner_loops=ops.iter_max;
        break;
    end 
    %}
end
details_LS.phi_diff=phi_diff;
end






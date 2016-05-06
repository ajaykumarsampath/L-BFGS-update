function [ alpha,details_LS ] = Goldstein_conditions( obj,Grad,Z,Y,d,ops)
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
% OUTPUT :     alpha  :  step size/media/ajay/New Volume/work/L-BFGS update
%

if(isfield(ops,'alpha'))
    alpha0=ops.alpha;
else
    alpha0=0;
end


tree=obj.SysMat_.tree;
sys=obj.SysMat_.sys;
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

[Zdir,Hessian_ops]=obj.Solve_step_direction(d);


Hx=ops.Hx;
Hx_term=ops.Hx_term;
T=ops.T;


b1=0;
b2=0;

b3.y=-(lambda*Hessian_ops.Hx+d.y);
b3.yt=cell(Ns,1);

b4.y=lambda*Hx+Y.y;
b4.yt=cell(Ns,1);

for j=1:non_leaf
    b1=b1+tree.prob(j)*Zdir.X(:,j)'*V.Q*Zdir.X(:,j)+tree.prob(j)*Zdir.U(:,j)'*...
        V.R*Zdir.U(:,j)+d.y(:,j)'*Hessian_ops.Hx(:,j)+0.5*lambda*norm(Hessian_ops.Hx(:,j))^2;
    
    b2=b2+2*tree.prob(j)*Zdir.X(:,j)'*V.Q*Z.X(:,j)+2*tree.prob(j)*Zdir.U(:,j)'*...
        V.R*Z.U(:,j)+d.y(:,j)'*Hx(:,j)+Y.y(:,j)'*Hessian_ops.Hx(:,j)+lambda*...
        Hx(:,j)'*Hessian_ops.Hx(:,j);
end 

for j=1:Ns
    b1=b1+tree.prob(non_leaf+j)*Zdir.X(:,non_leaf+j)'*V.Vf{j}*Zdir.X(:,non_leaf+j)+d.yt{j,1}'*...
        Hessian_ops.Hx_term{j,1}+0.5*lambda*norm(Hessian_ops.Hx_term{j,1})^2;
    
    b2=b2+2*tree.prob(non_leaf+j)*Zdir.X(:,non_leaf+j)'*V.Vf{j}*Z.X(:,non_leaf+j)+d.yt{j,1}'*...
        Hx_term{j,1}+Y.yt{j,1}'*Hessian_ops.Hx_term{j,1}+...
        lambda*Hessian_ops.Hx_term{j,1}'*Hx_term{j,1};
    
    b3.yt{j,1}=-lambda*Hessian_ops.Hx_term{j,1}-d.yt{j,1};
    b4.yt{j,1}=lambda*Hx_term{j,1}+Y.yt{j,1};
    
end

phi_cnst=vec(b4.y)'*vec(T.y)-0.5*lambda*norm(vec(T.y),2)^2;

for j=1:Ns
    phi_cnst=phi_cnst+b4.yt{j,1}'*T.yt{j,1}-0.5*lambda*norm(T.yt{j,1},2)^2;
end 
% phi_c the value function at the current direction
% alpha_p previous alpha
% alpah_c current alpha
% phi_p  value function at the previous direction



alpha_max=1;
% parameter for sufficient decrease condition
c1 = 1e-4;
% parameter for curvature condition
%c2 = 0.45;
c2 = 0.9;
alpha_ho=1;
alpha_lo=0;
alpha=0;
alpha_p=alpha0;
%alpha_c=rand(1)*alpha_max;
alpha_c=1;
phi_p=phi0;

i=1;
details_LS.term_LS=0;
details_LS.term_WF=0;
details_LS.inner_loops=0;
x0=ops.x0;
% difference in the objective functions  

while(abs(curv_dir)>1e-8)
    %% New direction--dual variable
    Ynew.y=Y.y+alpha_c*d.y;
    for j=1:Ns
        Ynew.yt{j,1}=Y.yt{j,1}+alpha_c*d.yt{j,1};
    end
    
   
    % new proximal value with g 
    Tnew.y=zeros(size(T.y));
    
    % difference in the objective functions  
    phi_diff=alpha_c^2*b1+alpha_c*b2+phi_cnst;
    
    for j=1:non_leaf
        % z(y+\tau d)
        Tnew.y(:,j)=min(lambda*Ynew.y(:,j)+Hx(:,j)+alpha_c...
            *Hessian_ops.Hx(:,j),sys.g{j});
        phi_diff=phi_diff-(b4.y(:,j)-alpha_c*b3.y(:,j))'*Tnew.y(:,j)...
            +0.5*lambda*norm(Tnew.y(:,j),2)^2;
    end
    
    for j=1:Ns
        % z(y+\tau d)
        Tnew.yt{j,1}=min(lambda*Y.yt{j,1}+Hx_term{j,1}+alpha_c*...
            Hessian_ops.Hx_term{j,1},sys.gt{j});
        phi_diff=phi_diff-(b4.yt{j,1}-alpha_c*b3.yt{j,1})'*Tnew.yt{j,1}...
            +0.5*lambda*norm(Tnew.yt{j,1},2)^2;
    end
    
    phi_diff=-phi_diff;
    
    % lower bound 
    YaLo.y=Y.y+alpha_lo*d.y;
    for j=1:Ns
        YaLo.yt{j,1}=Y.yt{j,1}+alpha_lo*d.yt{j,1};
    end
    phi_diff_lo=alpha_lo^2*b1+alpha_lo*b2+phi_cnst;
    
    for j=1:non_leaf
        % z(y+\tau d)
        Tlo.y(:,j)=min(lambda*YaLo.y(:,j)+Hx(:,j)+alpha_lo...
            *Hessian_ops.Hx(:,j),sys.g{j});
        phi_diff_lo=phi_diff_lo-(b4.y(:,j)-alpha_lo*b3.y(:,j))'*Tlo.y(:,j)...
            +0.5*lambda*norm(Tlo.y(:,j),2)^2;
    end
    
    for j=1:Ns
        % z(y+\tau d)
        Tlo.yt{j,1}=min(lambda*YaLo.yt{j,1}+Hx_term{j,1}+alpha_lo*...
            Hessian_ops.Hx_term{j,1},sys.gt{j});
        phi_diff_lo=phi_diff_lo-(b4.yt{j,1}-alpha_lo*b3.yt{j,1})'*Tlo.yt{j,1}...
            +0.5*lambda*norm(Tlo.yt{j,1},2)^2;
    end
    
    phi_diff_lo=-phi_diff_lo;
    %%
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
            *V.R*Znew.U(:,j)+0.5*lambda*(separ_var_new.y(:,j)'*...
            separ_var_new.y(:,j))+Ynew.y(:,j)'*separ_var_new.y(:,j);
        new_curv_dir=new_curv_dir+Grad_new.y(:,j)'*d.y(:,j);
    end
    
    for j=1:Ns
        phi_c=phi_c+tree.prob(non_leaf+j)*Znew.X(:,non_leaf+j)'*V.Vf{j}*Znew.X(:,non_leaf+j)+...
            Ynew.yt{j}'*separ_var_new.yt{j}+0.5*lambda*...
            norm(separ_var_new.yt{j})^2;
        new_curv_dir=new_curv_dir+Grad_new.yt{j}'*d.yt{j};
    end
    phi_c=-phi_c;
    
    %abs(new_curv_dir)<=-c2*curv_dir
    %}
    %%
    %{
    epsilon=1e-4;
    epsilon_inv=1e4;
    alpha_eps=alpha_c+epsilon;
    Yhess.y=Y.y+alpha_eps*d.y;
    for j=1:Ns
        Yhess.yt{j,1}=Y.yt{j,1}+alpha_eps*d.yt{j,1};
    end
    
   
    % new proximal value with g 
    Thess.y=zeros(size(T.y));
    
    % difference in the objective functions  
    phi_hess_diff=alpha_eps^2*b1+alpha_eps*b2+phi_cnst;
    
    for j=1:non_leaf
        % z(y+\tau d)
        Thess.y(:,j)=min(lambda*Yhess.y(:,j)+Hx(:,j)+alpha_eps...
            *Hessian_ops.Hx(:,j),sys.g{j});
        phi_hess_diff=phi_hess_diff-(b4.y(:,j)-alpha_eps*b3.y(:,j))'*Thess.y(:,j)...
            +0.5*lambda*norm(Thess.y(:,j),2)^2;
    end
    
    for j=1:Ns
        % z(y+\tau d)
        Thess.yt{j,1}=min(lambda*Yhess.yt{j,1}+Hx_term{j,1}+alpha_eps*...
            Hessian_ops.Hx_term{j,1},sys.gt{j});
        phi_hess_diff=phi_hess_diff-(b4.yt{j,1}-alpha_eps*b3.yt{j,1})'*Thess.yt{j,1}...
            +0.5*lambda*norm(Thess.yt{j,1},2)^2;
    end
    
    phi_hess_diff=-phi_hess_diff;
    
    error_grad=new_curv_dir-(phi_hess_diff-phi_diff)*epsilon_inv;
    
    percent_error_grad=(error_grad)/new_curv_dir
    %}
    
    %%
    if(phi_diff>c1*alpha_c*curv_dir || phi_diff-phi_diff_lo>0)
        alpha_ho=alpha_c;
    else
        if(phi_diff>c2*alpha_c*curv_dir)
        %if(abs(phi_diff-c2*alpha_c*curv_dir)<1e-6)
            alpha=alpha_c;
            details_LS.inner_loops=i;
            return
        else
            if(abs(alpha_c-1)<1e-4)
                alpha_lo=0;
            else
                alpha_lo=alpha_c;
            end 
            
        end    
    end
 
    alpha_temp(i)=alpha_c;
    i=i+1;
    if(i>ops.iter_max)
        disp('max iterations line search')
        alpha=alpha_c
        details_LS.term_LS=1;
        details_LS.inner_loops=ops.iter_max;
        return
    end
    
    alpha_c=(alpha_lo+alpha_ho)/2;
    %}
end

  
%{
    if(phi_diff<c1*alpha_c*curv_dir & phi_diff>c2*alpha_c*curv_dir)
        alpha=alpha_c;
        return
    else
        alpha_c=0.6*alpha_c;
    end
    %
    %}
end






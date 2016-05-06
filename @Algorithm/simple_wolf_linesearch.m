function [ alpha,details_LS ] = simple_wolf_linesearch( obj,Grad,Z,Y,d,ops)
%
% This function is the line search algorithm statisfying strong
% Wolfe conditions. Algorithm 3.5 on pages 60-61 in Nocedal and
% Wright. Here the line search is simplified for the scenario-tree problems
% with quadratic cost.
%
% Syntax : [ alpha ] = wolf_linesearch( fun_quad,Grad,x,d)
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
V=obj.SysMat_.V;
Nd=length(tree.stage);
Ns=length(tree.leaves);
non_leaf=Nd-Ns;
x0=ops.x0;


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

Zdir=obj.Solve_step_line_search(d);

phi0=-phi0;
%curv_dir=-curv_dir;
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
ops_zoom.Ns=Ns;
alpha=0;
alpha_p=alpha0;
alpha_c=rand(1)*alpha_max;
phi_p=phi0;

i=1;
details_LS.term_LS=0;
details_LS.term_WF=0;
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
    for j=1:Ns
        Ynew.yt{j,1}=Y.yt{j,1}+alpha_c*d.yt{j,1};
    end 
   
    % value function at Xkk
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
    %new_curv_dir=-new_curv_dir;
    
    if (phi_c > phi0+c1*alpha_c*curv_dir) || (phi_c >= phi_p && i>1)
        [alpha_k,term_WF]=obj.zoom_sectioning(Y,d,alpha_p,alpha_c,ops_zoom);
        details_LS.term_WF=term_WF; 
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
    
    if new_curv_dir>=0
        [alpha_k,term_WF]=obj.zoom_sectioning(Y,d,alpha_c,alpha_p,ops_zoom);
        alpha=alpha_k;
        details_LS.term_WF=term_WF; 
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
        details_LS.term_LS=1;
        return
    end
    %}
end


end








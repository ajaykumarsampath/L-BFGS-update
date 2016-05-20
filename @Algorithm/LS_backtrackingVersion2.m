function [ alpha,details_LS ] = LS_backtrackingVersion2(obj,Grad,Z,Y,d,ops)
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


Ytemp.y=Y.y+lambda*separ_vars.y;

for j=1:Ns
    Ytemp.yt{j,1}=Y.yt{j,1}+lambda*separ_vars.yt{j};
end

%
min_y=min(min(min(Ytemp.y)),min(min(cell2mat(Ytemp.yt))));
if(min_y<-1e-8)
    phi0=-inf;
    curv_dir=0;
else
    phi0=0;
    curv_dir=0;
    %temp_var=0;
    for j=1:non_leaf
        phi0=phi0+tree.prob(j)*Z.X(:,j)'*V.Q*Z.X(:,j)+tree.prob(j)*Z.U(:,j)'*V.R*Z.U(:,j)+...
            0.5*lambda*norm(separ_vars.y(:,j))^2+Y.y(:,j)'*separ_vars.y(:,j)...
            -ops.g(:,j)'*Ytemp.y(:,j)+ops.T.y(:,j)'*Ytemp.y(:,j);
        curv_dir=curv_dir+Grad.y(:,j)'*d.y(:,j);
        %temp_var=temp_var-ops.g(:,j)'*Ytemp.y(:,j)+ops.T.y(:,j)'*Ytemp.y(:,j);
    end
    
    for j=1:Ns
        phi0=phi0+tree.prob(non_leaf+j)*Z.X(:,non_leaf+j)'*V.Vf{j}*Z.X(:,non_leaf+j)+...
            Y.yt{j}'*separ_vars.yt{j}+0.5*lambda*norm(separ_vars.yt{j})^2....
            -ops.gt{j,1}'*Ytemp.yt{j,1}+ops.T.yt{j,1}'*Ytemp.yt{j};
        curv_dir=curv_dir+Grad.yt{j}'*d.yt{j};
    end
    phi0=-phi0;
    
end
 %}


% phi0,phi_c are the value function at the current and previous direction
% phi_diff=phi_c-phi0

alpha_c=lambda;
%alpha_c=1;
beta=0.5;
i=1;
details_LS.term_LS=0;
details_LS.inner_loops=0;
x0=ops.x0;
% difference in the objective functions  

while(abs(curv_dir)>1e-8)
    %% New direction--dual variable
    W.y=Y.y+alpha_c*d.y;
    for j=1:Ns
        W.yt{j,1}=Y.yt{j,1}+alpha_c*d.yt{j,1};
    end
    
    % Envelope gradient
    [Grad_w,Zw,details_w] =obj.grad_dual_envelop(W,x0);
    
    separ_var_new.y=details_w.Hx-details_w.T.y;
    % 
    Wtemp.y=W.y+details_w.lambda*separ_var_new.y;
    
    for j=1:Ns
        separ_var_new.yt{j,1}=details_w.Hx_term{j,1}-details_w.T.yt{j,1};
        Wtemp.yt{j,1}=W.yt{j,1}+details_w.lambda*separ_var_new.yt{j,1};
    end
    
    min_y=min(min(min(Wtemp.y)),min(min(cell2mat(Wtemp.yt))));
   
    
    %
    if(min_y<-1e-6)
        phi_c=inf;
        new_curv_dir=0;
    else
        phi_c=0;
        new_curv_dir=0;
        temp_var=0;
        for j=1:non_leaf
            phi_c=phi_c+tree.prob(j)*Zw.X(:,j)'*V.Q*Zw.X(:,j)+tree.prob(j)*Zw.U(:,j)'...
                *V.R*Zw.U(:,j)+0.5*details_w.lambda*(separ_var_new.y(:,j)'*...
                separ_var_new.y(:,j))+W.y(:,j)'*separ_var_new.y(:,j)...
                -ops.g(:,j)'*Wtemp.y(:,j)+details_w.T.y(:,j)'*Wtemp.y(:,j);
            new_curv_dir=new_curv_dir+Grad_w.y(:,j)'*d.y(:,j);
            temp_var=temp_var-ops.g(:,j)'*Wtemp.y(:,j)+details_w.T.y(:,j)'*Wtemp.y(:,j);
        end
        
        for j=1:Ns
            phi_c=phi_c+tree.prob(non_leaf+j)*Zw.X(:,non_leaf+j)'*V.Vf{j}*Zw.X(:,non_leaf+j)+...
                W.yt{j}'*separ_var_new.yt{j}+0.5*details_w.lambda*norm(separ_var_new.yt{j})^2....
                -ops.gt{j,1}'*Wtemp.yt{j,1}+details_w.T.yt{j,1}'*Wtemp.yt{j};
            temp_var=temp_var-ops.gt{j,1}'*Wtemp.yt{j,1}+details_w.T.yt{j,1}'*Wtemp.yt{j};
            new_curv_dir=new_curv_dir+Grad_w.yt{j}'*d.yt{j};
        end
        phi_c=-phi_c;
    end
    
    if(temp_var<-1e-6)
        error('mistake')
    end
    %}
    
    %
    % cost of the FBS
    Ztemp=obj.Solve_step(Wtemp,x0);
    [Y2,details_temp_prox]=obj.GobalFBS_proximal_gcong(Ztemp,Wtemp);
    
    prm_infs.y=details_temp_prox.Hx-details_temp_prox.T.y;
    for j=1:Ns
        prm_infs.yt{j}=details_temp_prox.Hx_term{j}-details_temp_prox.T.yt{j};
    end
    
    phi_temp=0;
    for j=1:non_leaf
        phi_temp=phi_temp+tree.prob(j)*Ztemp.X(:,j)'*V.Q*Ztemp.X(:,j)...
            +tree.prob(j)*Ztemp.U(:,j)'*V.R*Ztemp.U(:,j)+Wtemp.y(:,j)'*prm_infs.y(:,j);
    end
    
    for j=1:Ns
        phi_temp=phi_temp+tree.prob(non_leaf+j)*Ztemp.X(:,non_leaf+j)'*...
            V.Vf{j}*Ztemp.X(:,non_leaf+j)+Wtemp.yt{j}'*prm_infs.yt{j};
    end
    
    phi_temp=-phi_temp;
    details_LS.env_diff=phi_temp-phi_c;
    %}
    
    phi_diff=phi_c-phi0;
    %if(phi_diff>0 || details_LS.env_diff>0)
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
details_LS.phi_c=phi_c;
details_LS.phi0=phi0;
details_LS.Znew=Zw;
details_LS.Hx=details_w.Hx;
details_LS.T=details_w.T;
details_LS.lambda=details_w.lambda;
%details_LS.W=Y2;

end






function [Z,Y1,details]=Dual_FBE(obj,x0)
% This function calculate the optimal solution using the
% APG algorithm on the the dual problem for the system at the
% given initial point
%
% INPUT:     x0  :  initial point
%
% OUTPUT:    Z   :  primal solution
%            Y   :  dual solution
%       details  :  structure containing details like time of
%                   computation, number of itrataions.
%

sys=obj.SysMat_.sys;
tree=obj.SysMat_.tree;
V=obj.SysMat_.V;
ops=obj.algo_details.ops_FBE;

Ns=length(tree.leaves); % total scenarios in the tree
Nd=length(tree.stage); %  toal nodes in the tree
non_leaf=Nd-Ns;
%lambda=obj.algo_details.ops_FBE.lambda;


% options for the line search
ops_step_size.x0=x0;
ops_step_size.iter_max=10;
% Initalizing the dual varibables
Y0.y=zeros(size(sys.F{1},1),Nd-Ns);
Y1.y=zeros(size(sys.F{1},1),Nd-Ns);

prm_fes_term=cell(Ns,1);
epsilon_prm=1;

for i=1:Ns
    Y0.yt{i,:}=zeros(size(sys.Ft{i,1},1),1);
    Y1.yt{i,:}=zeros(size(sys.Ft{i,1},1),1);
    prm_fes_term{i,1}=zeros(size(sys.Ft{i,1},1),1);
end

%prm_fes=zeros(size(sys.F{1},1),Nd-Ns);
g_nodes=zeros(size(sys.F{1},1),Nd-Ns);
for i=1:Nd-Ns
    g_nodes(:,i)=sys.g{i};
end
%g_nodes_term=sys.gt;
theta=[1 1]';
tic
j=1;

W_minyt=zeros(Ns,1);

details.term_crit=zeros(1,4);
%dual_grad=prm_fes;
%dual_grad_term=prm_fes_term;

while(j<ops.steps)
    % Step 1: accelerated step
    W.y=Y1.y+theta(2)*(1/theta(1)-1)*(Y1.y-Y0.y);
    
    for i=1:Ns
        W.yt{i,1}=Y1.yt{i,1}+theta(2)*(1/theta(1)-1)*(Y1.yt{i,1}-Y0.yt{i,1});
    end
    
    % step 2 : evaluation of the gradient of envelop.
    if(j==1)
        Y0.y=Y1.y;
        Y0.yt=Y1.yt;
        [Grad_env,Z,details_prox] =obj.grad_dual_envelop(W,x0);
        lambda=details_prox.lambda;
        obj.algo_details.ops_FBE.lambda=details_prox.lambda;
        details.lambda_prox(j)=details_prox.lambda;
        
        Y1.y=W.y+lambda*(details_prox.Hx-details_prox.T.y);
        for i=1:Ns
            Y1.yt{i}=W.yt{i}+lambda*(details_prox.Hx_term{i}-details_prox.T.yt{i});
        end
        ops_step_size.separ_vars.y=details_prox.Hx-details_prox.T.y;
        for i=1:Ns
            ops_step_size.separ_vars.yt{i}=details_prox.Hx_term{i}-details_prox.T.yt{i};
        end
        details_LS.term_LS=0;
        details_LS.term_WF=0;
        
    else
%%  
        %obj.algo_details.ops_FBE.lambda
        [Grad_env,Z,details_prox] =obj.grad_dual_envelop(W,x0);   
        obj.algo_details.ops_FBE.lambda=details_prox.lambda;
        details.lambda_prox(j)=details_prox.lambda;
        %details.pos_def(j)=details_prox.pos_def;
        %obj1=obj;
        % calculate the direction by LBFGS method
        [obj,dir_env]= obj.LBFGS_direction(Grad_env,Grad_envOld,W,Wold);
        details.H(j)=obj.algo_details.ops_FBE.Lbfgs.H;
        details.Hnum(j)=obj.algo_details.ops_FBE.Lbfgs.Hnum;
        details.Hden(j)=obj.algo_details.ops_FBE.Lbfgs.Hden;
        details.YS(j)=obj.algo_details.ops_FBE.Lbfgs.rho;
        %details.Grad_env{j}=Grad_env;
        %details.dir_env{j}=dir_env;
        details.direction(j)=0;
        for i=1:non_leaf
            details.direction(j)=details.direction(j)+dir_env.y(:,i)'*Grad_env.y(:,i);
        end
        for i=1:Ns
            details.direction(j)=details.direction(j)+dir_env.yt{i}'*Grad_env.yt{i};
        end
        
        if(abs(details.direction(j))<1e-3)
            details.inner_loops(j,1)=0;
            break
        end
        % step size calcualtion
        ops_step_size.separ_vars.y=details_prox.Hx-details_prox.T.y;
        for i=1:Ns
            ops_step_size.separ_vars.yt{i}=details_prox.Hx_term{i}-details_prox.T.yt{i};
        end
        %
        if(strcmp(obj.algo_details.ops_FBE.LS,'WOLFE'))
            [details.tau(j),details_LS]=obj.wolf_linesearch(Grad_env,Z,W,dir_env,ops_step_size);
            details.inner_loops(j,1)=details_LS.inner_iter;
            
            %details.lambda(j)=lambda;
            %details_LS.term_LS=0;
            %details_LS.term_WF=0;
        else
            ops_step_size.Hx=details_prox.Hx;
            ops_step_size.Hx_term=details_prox.Hx_term;
            ops_step_size.T=details_prox.T;
            [details.lambda(j),details_LS]=obj.Goldstein_conditions...
                (Grad_env,Z,W,dir_env,ops_step_size);
            details.inner_loops(j,1)=details_LS.inner_loops;
        end

        
        Y0.y=Y1.y;
        Y0.yt=Y1.yt;
        if(details_LS.term_LS || details_LS.term_WF)
            details_LS
        end
        %{
        lambda=obj.algo_details.ops_FBE.lambda;
        Y1.y=W.y+lambda*(details_prox.Hx-details_prox.T.y);
        for i=1:Ns
           Y1.yt{i}=W.yt{i}+lambda*(details_prox.Hx_term{i}-details_prox.T.yt{i});
        end
        %}
        %lambda=obj.algo_details.ops_FBE.lambda;
        if(~details_LS.term_LS && ~details_LS.term_WF)
            p=1;
            %lambda=0.5;
            %p=1;
            tau=details.tau(j);
            Y1.y=W.y+p*tau*dir_env.y;
            for i=1:Ns
                Y1.yt{i}=W.yt{i}+p*tau*dir_env.yt{i};
            end
        end
        
%%       
        %}
    end
    %% 
    details.cost_function(j)=0;
    for i=1:non_leaf
        details.cost_function(j)=details.cost_function(j)+tree.prob(i)*Z.X(:,i)'*V.Q*Z.X(:,i)...
            +tree.prob(i)*Z.U(:,i)'*V.R*Z.U(:,i);
        %+0.5*lambda*(ops_step_size.separ_vars.y(:,i)'*...
         %ops_step_size.separ_vars.y(:,i))+Y1.y(:,i)'*ops_step_size.separ_vars.y(:,i);
    end
    
    for i=1:Ns
        details.cost_function(j)=details.cost_function(j)+tree.prob(non_leaf+i)*Z.X(:,non_leaf+i)'...
            *V.Vf{i}*Z.X(:,non_leaf+i);
        %+Y1.yt{i}'*ops_step_size.separ_vars.yt{i}+0.5*lambda*norm(ops_step_size.separ_vars.yt{i})^2;
    end
    %{
    term=0;
    for i=1:non_leaf
        term=term+Y1.y(:,i)'*ops_step_size.separ_vars.y(:,i);
    end 
    for i=1:Ns
        term=term+Y1.yt{i}'*ops_step_size.separ_vars.yt{i};
    end
    term
    %}
    %details.cost_function(j)=details.cost_function(j);
    Grad_envOld=Grad_env;
    Wold=W;
    
   
    %max(max(ops_step_size.separ_vars.y))
    if(norm(ops_step_size.separ_vars.y,inf)<ops.primal_inf || details_LS.term_LS || details_LS.term_WF)
        details.iter=j;
        obj.algo_details.ops_FBE.Lbfgs;
        break
    else
        %theta(1)=theta(2);
        %theta(2)=(sqrt(theta(1)^4+4*theta(1)^2)-theta(1)^2)/2;
        j=j+1;
    end
   
    %%
end

details.dual_gap=0;
for i=1:non_leaf
    details.dual_gap=details.dual_gap+Y1.y(:,i)'*ops_step_size.separ_vars.y(:,i);
end

for i=1:Ns
    details.dual_gap=details.dual_gap+Y1.yt{i}'*ops_step_size.separ_vars.yt{i};
end

%obj.algo_details.ops_FBE.Lbfg
details.iter=j;
details.separ_var.y=(ops_step_size.separ_vars.y);
details.separ_var.yt=ops_step_size.separ_vars.yt;
details.Hx=details_prox.Hx;
details.T=details_prox.T;
details.FBE_solve=toc;
details.W=W;



end
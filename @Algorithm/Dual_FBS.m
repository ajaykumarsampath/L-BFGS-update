function [Z,Y1,details]=Dual_FBS(obj,x0)
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
Tree=obj.SysMat_.tree;
V=obj.SysMat_.V;
ops=obj.algo_details.ops_APG;

Ns=length(Tree.leaves); % total scenarios in the tree
Nd=length(Tree.stage); %  toal nodes in the tree
non_leaf=Nd-Ns;
lambda=obj.algo_details.ops_FBS.lambda;

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
        W_minyt(i,1)=min(W.yt{i,1});
    end
    
    
    % step 2 : evaluation of the gradient of envelop.
    if(j==1)
        Y0.y=Y1.y;
        Y0.yt=Y1.yt;
        [Grad_env,Z,details_prox] =obj.grad_dual_envelop(W,x0);
        Y1.y=W.y+lambda*(details_prox.Hx-details_prox.T.y);
        for i=1:Ns
            Y1.yt{i}=W.yt{i}+lambda*(details_prox.Hx_term{i}-details_prox.T.yt{i});
        end
        ops_step_size.separ_vars.y=details_prox.Hx-details_prox.T.y;
        for i=1:Ns
            ops_step_size.separ_vars.yt{i}=details_prox.Hx_term{i}-details_prox.T.yt{i};
        end
    else
        [Grad_env,Z,details_prox] =obj.grad_dual_envelop(W,x0);
        
        % calculate the direction by LBFGS method
        [obj,dir_env]= obj.LBFGS_direction(Grad_env,Grad_envOld,W,Wold);
        details.direction(j)=0;
        for i=1:non_leaf
            details.direction(j)=details.direction(j)+dir_env.y(:,i)'*Grad_env.y(:,i);
        end 
        for i=1:Ns
            details.direction(j)=details.direction(j)+dir_env.yt{i}'*Grad_env.yt{i};
        end 
        % step size calcualtion
        ops_step_size.separ_vars.y=details_prox.Hx-details_prox.T.y;
        for i=1:Ns
            ops_step_size.separ_vars.yt{i}=details_prox.Hx_term{i}-details_prox.T.yt{i};
        end
        
        Y0.y=Y1.y;
        Y0.yt=Y1.yt;
        %{
        Y1.y=W.y+lambda*(details_prox.Hx-details_prox.T.y);
        for i=1:Ns
            Y1.yt{i}=W.yt{i}+lambda*(details_prox.Hx_term{i}-details_prox.T.yt{i});
        end
        %}
        %
        if(mod(j,10)>0)
            p=0.001;
        else
            p=0.01;
        end 
        Y1.y=W.y-p*lambda*dir_env.y;
        for i=1:Ns
            Y1.yt{i}=W.yt{i}-p*lambda*dir_env.yt{i};
        end
        %}
    end
    %}
    
    Grad_envOld=Grad_env;
    Wold=W;
    
    if(max(ops_step_size.separ_vars.y)<0.01)
        details.iter=j;
        return
    else
        %theta(1)=theta(2);
        %theta(2)=(sqrt(theta(1)^4+4*theta(1)^2)-theta(1)^2)/2;
        j=j+1;
    end
end
details.iter=j;
details.gpad_solve=toc;
details.W=W;
%details.epsilon_prm_avg= epsilon_prm_avg;
%details.epsilon_prm=epsilon_prm;


end
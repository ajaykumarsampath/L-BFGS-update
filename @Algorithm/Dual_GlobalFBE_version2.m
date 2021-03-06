function [Z,Y,details]=Dual_GlobalFBE_version2(obj,x0)
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
ops_step_size.iter_max=5;
ops_step_size.g=cell2mat(sys.g);
ops_step_size.gt=sys.gt;
% Initalizing the dual varibables
Y.y=zeros(size(sys.F{1},1),Nd-Ns);

prm_fes_term=cell(Ns,1);

for i=1:Ns
    Y.yt{i,:}=zeros(size(sys.Ft{i,1},1),1);
    prm_fes_term{i,1}=zeros(size(sys.Ft{i,1},1),1);
end

%prm_fes=zeros(size(sys.F{1},1),Nd-Ns);
g_nodes=zeros(size(sys.F{1},1),Nd-Ns);
for i=1:Nd-Ns
    g_nodes(:,i)=sys.g{i};
end
%g_nodes_term=sys.gt;
tic
j=1;

details.term_crit=zeros(1,4);
%memory=obj.algo_details.ops_FBE.memory;

grad_steps=0;
Lbfgs_loop=0;

%MatLBFG=obj.algo_details.ops_FBE.Lbfgs;
 
while(j<ops.steps)
    
    % Step 1: accelerated step
    W=Y;
    
    % step 2 : evaluation of the gradient of envelope
    [Grad_env,Zint,details_prox] =obj.grad_dual_envelop(W,x0);
    obj.algo_details.ops_FBE.lambda=details_prox.lambda;
    details.lambda_prox(j)=details_prox.lambda;
    
    if(Lbfgs_loop)
        % calculate a new direction the quasi-newton method--LBFGS
        if(j==4)
            obj1=obj;
        end
        
        %MatLBFG=obj.algo_details.ops_FBE.Lbfgs;
        
        [MatLBFG_temp,dir_env]= obj.LBFGS_direction_version2(Grad_env,Grad_envOld,W,Wold);
        details.H(j)=MatLBFG_temp.H;
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
        
        
        %
        ops_step_size.separ_vars.y=details_prox.Hx-details_prox.T.y;
        for i=1:Ns
            ops_step_size.separ_vars.yt{i}=details_prox.Hx_term{i}-details_prox.T.yt{i};
        end
        ops_step_size.T=details_prox.T;
        %}
        if(details.H(j)>0)
            [details.tau(j),details_LS]=obj.LS_backtrackingVersion2(Grad_env,Zint,W,dir_env,...
                ops_step_size);
            details.inner_loops(j,1)=details_LS.inner_loops;
            details.phi_diff(j,1)=details_LS.phi_diff;
            %details.env_diff(j,1)=details_LS.env_diff;
            %if(details.env_diff(j,1)>0)
             %   details.tau(j)=0;
            %end
        else
            %j
            details_LS.term_LS=1;
            details.tau(j)=0;
            details.inner_loops(j,1)=0;
            details.phi_diff(j,1)=0;
        end 
        %if(details.tau(j)>0)
            obj.algo_details.ops_FBE.Lbfgs=MatLBFG_temp;
            Wold=W;
            Grad_envOld=Grad_env;
            tau=details.tau(j);
            W.y=W.y+tau*dir_env.y;
            for i=1:Ns
                W.yt{i}=W.yt{i}+tau*dir_env.yt{i};
            end
        %else
         %   obj.algo_details.ops_FBE.Lbfgs=MatLBFG;
            %Wold=W;
            %Grad_envOld=Grad_env;
        %end
        if(details_LS.term_LS)
            
            %obj.algo_details.ops_FBE.Lbfgs=Lbfgs_ops;
            grad_steps=grad_steps+1;
            %skip_step=1;
            %{
             %if(grad_steps==0)
                 memory=obj.algo_details.ops_FBE.memory;
                 dim_dual=2*(non_leaf*(sys.nx+sys.nu)+Ns*sys.nx);
                 obj.algo_details.ops_FBE.Lbfgs.S=zeros(dim_dual, memory); % dual variable
                 obj.algo_details.ops_FBE.Lbfgs.Y  = zeros(dim_dual,memory); % dual gradient
                 obj.algo_details.ops_FBE.Lbfgs.YS = zeros(memory, 1);
                 obj.algo_details.ops_FBE.Lbfgs.LBFGS_col = 1;
                 obj.algo_details.ops_FBE.Lbfgs.LBFGS_mem = 0;
                 obj.algo_details.ops_FBE.Lbfgs.skipCount = 0;
             %end
            %}
        end
    end
    
    %step 3: dual gradient calculation
    Z=obj.Solve_step(W,x0);
    
    %step 4 gradient projection algorithm
    [Y,details_prox]=obj.GobalFBS_proximal_gcong(Z,W);
    
    min_y=min(min(min(Y.y)),min(min(cell2mat(Y.yt))));
    
    if(min_y<-1e-8)
        min_y
    end 
    obj.algo_details.ops_FBE.lambda=details_prox.lambda;
    
    if(j==1)
        Grad_envOld=Grad_env;
        Wold=W;
        Lbfgs_loop=1;
    end
    
    %{
    for i=1:non_leaf
        Y1.y(:,i)=W.y(:,i)+details_prox.lambda*(details_prox.Hx(:,i)-details_prox.T.y(:,i));
    end
    
    for i=1:Ns
        Y1.yt{i,1}=W.yt{i,1}+details_prox.lambda*(details_prox.Hx_term{i,1}-details_prox.T.yt{i,1});
    end
    
    %max(max(Y.y-Y1.y))
    %}
    prm_infs.y=details_prox.Hx-details_prox.T.y;
    for i=1:Ns
        prm_infs.yt{i}=details_prox.Hx_term{i}-details_prox.T.yt{i};
    end
    
    details.cost_function(j)=0;
    for i=1:non_leaf
        details.cost_function(j)=details.cost_function(j)+tree.prob(i)*Z.X(:,i)'*V.Q*Z.X(:,i)...
            +tree.prob(i)*Z.U(:,i)'*V.R*Z.U(:,i);%...
        %+0.5*details_prox.lambda*(prm_infs.y(:,i)'*prm_infs.y(:,i))+Y.y(:,i)'*prm_infs.y(:,i);
    end
    
    for i=1:Ns
        details.cost_function(j)=details.cost_function(j)+tree.prob(non_leaf+i)*Z.X(:,non_leaf+i)'...
            *V.Vf{i}*Z.X(:,non_leaf+i);%...
        %+Y.yt{i}'*prm_infs.yt{i}+0.5*details_prox.lambda*norm(prm_infs.yt{i})^2;
    end
    %max(max(ops_step_size.separ_vars.y))
    if(max(max(abs(prm_infs.y)))<0.01)
        details.iter=j;
        obj.algo_details.ops_FBE.Lbfgs;
        break
    else
        %theta(1)=theta(2);
        %theta(2)=(sqrt(theta(1)^4+4*theta(1)^2)-theta(1)^2)/2;
        j=j+1;
    end
    
end 

details.dual_gap=0;
for i=1:non_leaf
    details.dual_gap=details.dual_gap+Y.y(:,i)'*prm_infs.y(:,i);
end

for i=1:Ns
    details.dual_gap=details.dual_gap+Y.yt{i}'*prm_infs.yt{i};
end

%obj.algo_details.ops_FBE.Lbfg
details.iter=j;
details.prm_infs=prm_infs;
details.Hx=details_prox.Hx;
details.T=details_prox.T;
details.grad_steps=grad_steps;
details.FBE_solve=toc;
details.W=W;


end



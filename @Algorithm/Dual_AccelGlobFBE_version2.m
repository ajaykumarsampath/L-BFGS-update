function [Z,Y0,details]=Dual_AccelGlobFBE_version2(obj,x0)
%
% This function calculate the optimal solution using the
% APG algorithm with an intermediate step of L-BFGS update on the 
% Forward-Backward Envelope (FBE) constructed for the dual 
% formulation for the system at an initial point 
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
% Initalizing the dual varibables
Y0.y=zeros(size(sys.F{1},1),Nd-Ns);

prm_fes_term=cell(Ns,1);

for i=1:Ns
    Y0.yt{i,1}=zeros(size(sys.Ft{i,1},1),1);
    prm_fes_term{i,1}=zeros(size(sys.Ft{i,1},1),1);
end

Y1=Y0;
S=Y0;
theta=[1 1];
%prm_fes=zeros(size(sys.F{1},1),Nd-Ns);
g_nodes=zeros(size(sys.F{1},1),Nd-Ns);
for i=1:Nd-Ns
    g_nodes(:,i)=sys.g{i};
end

tic
j=1;

details.term_crit=zeros(1,4);
%memory=obj.algo_details.ops_FBE.memory;

grad_steps=0;
Lbfgs_loop=0;

% cost on the envelope function
phi_prev=inf;

while(j<ops.steps)
    
    % Step 1: accelerated step
    %U.y=Y1.y+theta(2)*(Y0.y-Y1.y)-theta(2)/theta(1)*(Y0.y-S.y);
    U.y=Y1.y+theta(2)*(Y0.y-Y1.y)-theta(2)/theta(1)*(Y0.y-S.y);
    
    for i=1:Ns
        U.yt{i,1}=Y1.yt{i,1}+theta(2)*(Y0.yt{i,1}-Y1.yt{i,1})...
            -theta(2)/theta(1)*(Y0.yt{i,1}-S.yt{i,1});
    end 
    
    % Step 2: monotonicity on the envelope.
    % First calculate the gradient of envelope
    [Grad_env,Zint,details_prox] =obj.grad_dual_envelop(U,x0); 
    obj.algo_details.ops_FBE.lambda=details_prox.lambda;
    details.lambda_prox(j)=details_prox.lambda;
    phi_cur=details_prox.phi;
     
    % Update with the proximal algorithm 
    S.y=U.y+details_prox.lambda*(details_prox.Hx-details_prox.T.y);
    for i=1:Ns
        S.yt{i,1}=U.yt{i,1}+details_prox.lambda*(details_prox.Hx_term{i,1}...
            -details_prox.T.yt{i,1});
    end 
    %
    if(strcmp(obj.algo_details.ops_FBE.monotonicity,'yes'))
        if(phi_cur>phi_prev)
            U=Y1;
            %[j phi_cur-phi_prev]
            [Grad_env,Zint,details_prox] =obj.grad_dual_envelop(U,x0);
            obj.algo_details.ops_FBE.lambda=details_prox.lambda;
            details.lambda_prox(j)=details_prox.lambda;
            phi_cur=details_prox.phi;
        end
    end
    %}
    if(Lbfgs_loop)
        % calculate a new direction the quasi-newton method--LBFGS
        
        [obj,dir_env]= obj.LBFGS_direction(Grad_env,Grad_envOld,U,Uold);
        details.H(j)=obj.algo_details.ops_FBE.Lbfgs.H;
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
        
        ops_step_size.separ_vars.y=details_prox.Hx-details_prox.T.y;
        for i=1:Ns
            ops_step_size.separ_vars.yt{i}=details_prox.Hx_term{i}-details_prox.T.yt{i};
        end
        if(details.H(j)>0)
            [details.tau(j),details_LS]=obj.LS_backtrackingVersion2...
                (Grad_env,Zint,U,dir_env,ops_step_size);
            details.inner_loops(j,1)=details_LS.inner_loops;
            details.phi_diff(j,1)=details_LS.phi_diff;

        else
            details.tau(j)=0;
            details.inner_loops(j,1)=0;
            details.phi_diff(j,1)=0;
  
        end 
        
        Uold=U;
        Grad_envOld=Grad_env;
        tau=details.tau(j);
        W.y=U.y+tau*dir_env.y;
        for i=1:Ns
            W.yt{i,1}=U.yt{i,1}+tau*dir_env.yt{i,1};
        end
        if(details_LS.term_LS)
            
            grad_steps=grad_steps+1;
            
        end
        
    end
    
    % updating the dual vector
    Y0.y=Y1.y;
    Y0.yt=Y1.yt;
    
    if(j==1)
        % step 3: dual gradient calculation 
        Z=Zint;
        % step 4: gradient projection algorithm
        % [Y1,details_prox]=obj.GobalFBS_proximal_gcong(Z,U);
        % obj.algo_details.ops_FBE.lambda=details_prox.lambda;
        
        Y1=S;
        Uold=U;
        Grad_envOld=Grad_env;
        Lbfgs_loop=1;
    else
        % step 3: dual gradient calculation at the Envelope variable
        Z=obj.Solve_step(W,x0);
        
        % step 4: gradient projection algorithm
        [Y1,details_prox]=obj.GobalFBS_proximal_gcong(Z,W);
        obj.algo_details.ops_FBE.lambda=details_prox.lambda;
    end
    
    
    % calculating the primal infeasibility 
    prm_infs.y=details_prox.Hx-details_prox.T.y;
    for i=1:Ns
        prm_infs.yt{i}=details_prox.Hx_term{i}-details_prox.T.yt{i};
    end
    
    % value function on the envelope  
    if(j>1)
        phi_prev=0;
        for i=1:non_leaf
            phi_prev=phi_prev+tree.prob(i)*Z.X(:,i)'*V.Q*Z.X(:,i)+tree.prob(i)*Z.U(:,i)'*...
                V.R*Z.U(:,i)+0.5*details_prox.lambda*norm(prm_infs.y(:,i))^2+W.y(:,i)'*prm_infs.y(:,i);
        end
        
        for i=1:Ns
            phi_prev=phi_prev+tree.prob(non_leaf+i)*Z.X(:,non_leaf+i)'*V.Vf{i}*Z.X(:,non_leaf+i)+...
                W.yt{i,1}'*prm_infs.yt{i}+0.5*details_prox.lambda*norm(prm_infs.yt{i})^2;
        end
        phi_prev=-phi_prev;
        %[details.phi_c(j,1)-phi_prev details.phi0(j,1)-phi_cur phi_prev-phi_cur-details.phi_diff(j,1)]
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
    epsilon=max(max(abs(prm_infs.y)));
    %epsilon=max(max(max(abs(cell2mat(prm_infs.yt)))),epsilon);
    if(epsilon<0.01)
        details.iter=j;
        obj.algo_details.ops_FBE.Lbfgs;
        break
    else
        theta(1)=theta(2);
        theta(2)=(sqrt(theta(1)^4+4*theta(1)^2)-theta(1)^2)/2;
        j=j+1;
    end
    
end

details.dual_gap=0;
for i=1:non_leaf
    details.dual_gap=details.dual_gap+Y0.y(:,i)'*prm_infs.y(:,i);
end

for i=1:Ns
    details.dual_gap=details.dual_gap+Y0.yt{i}'*prm_infs.yt{i};
end

%obj.algo_details.ops_FBE.Lbfg
details.iter=j;
details.prm_infs=prm_infs;
details.Hx=details_prox.Hx;
details.T=details_prox.T;
details.grad_steps=grad_steps;
details.FBE_solve=toc;
details.Y=Y1;

%{
    % calculating the primal infeasibility
    prm_infs.y=details_prox.Hx-details_prox.T.y;
    for i=1:Ns
        prm_infs.yt{i}=details_prox.Hx_term{i}-details_prox.T.yt{i};
    end
    
    % value function on the envelope
    
    phi_cur1=0;
    for i=1:non_leaf
        phi_cur1=phi_cur1+tree.prob(i)*Zint.X(:,i)'*V.Q*Zint.X(:,i)+tree.prob(i)*Zint.U(:,i)'*...
            V.R*Zint.U(:,i)+0.5*details_prox.lambda*norm(prm_infs.y(:,i))^2+U.y(:,i)'*prm_infs.y(:,i);
    end
    
    for i=1:Ns
        phi_cur1=phi_cur1+tree.prob(non_leaf+i)*Zint.X(:,non_leaf+i)'*V.Vf{i}*Zint.X(:,non_leaf+i)+...
            U.yt{i,1}'*prm_infs.yt{i}+0.5*details_prox.lambda*norm(prm_infs.yt{i})^2;
    end
    phi_cur1=-phi_cur1;
    
    phi_cur-phi_cur1

    details.cost_function2(j)=0;
    for i=1:non_leaf
        details.cost_function2(j)=details.cost_function2(j)+tree.prob(i)*Zint.X(:,i)'*V.Q*Zint.X(:,i)...
            +tree.prob(i)*Zint.U(:,i)'*V.R*Zint.U(:,i);
    end
    
    for i=1:Ns
        details.cost_function2(j)=details.cost_function2(j)+tree.prob(non_leaf+i)*Zint.X(:,non_leaf+i)'...
            *V.Vf{i}*Zint.X(:,non_leaf+i);
    end
            %details.phi_c(j,1)=details_LS.phi_c;
            %details.phi0(j,1)=details_LS.phi0;
          %details.phi_c(j,1)=0;
            %details.phi0(j,1)=0;
%}


end





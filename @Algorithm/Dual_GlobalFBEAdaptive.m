function [Z,Y,details]=Dual_GlobalFBEAdaptive(obj,x0)
% This function calculate the optimal solution using the
% APG algorithm on the the dual problem for the system at the
% given initial point
%
% Version: Implements the line search with the same lambda; 
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
nx=sys.nx;
nu=sys.nu;
nz=(nx+nu);
ny=2*nz;
ops.primal_inf=obj.algo_details.ops_FBE.primal_inf;
beta=obj.algo_details.ops_FBE.betaB;
%lambda=obj.algo_details.ops_FBE.lambda;


% options for the line search
ops_step_size.x0=x0;
ops_step_size.iter_max=5;
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

grad_steps=0;
Lbfgs_loop=0;
%%
while(j<ops.steps)
    
    % Step 1: accelerated step
    W=Y;
%     if(j==11)
%         j
%     end
    if(Lbfgs_loop)
        kk=0;
        lambda=obj.algo_details.ops_FBE.lambda;
        while(kk<5)
            % step 2 : evaluation of the gradient of envelope
            [Grad_env,Zint,details_env] =obj.grad_dual_envelopVersion2(W,x0);
            %obj.algo_details.ops_FBE.lambda=details_env.lambda;
            
            % calculate a new direction the quasi-newton method--LBFGS
            %Lbfgs_col=obj.algo_details.ops_FBE.Lbfgs.LBFGS_col;
            MLBFGS=obj.algo_details.ops_FBE.Lbfgs;
            [obj,dir_env]= obj.LBFGS_direction(Grad_env,Grad_envOld,W,Wold);
            details.H(j)=obj.algo_details.ops_FBE.Lbfgs.H;
            details.direction(j)=0;

            for i=1:non_leaf
                details.direction(j)=details.direction(j)+dir_env.y(:,i)'*Grad_env.y(:,i);
            end
            
            for i=1:Ns
                details.direction(j)=details.direction(j)+dir_env.yt{i}'*Grad_env.yt{i};
            end
            
            if(abs(details.direction(j))<1e-6)
                details.inner_loops(j,1)=0;
                tau=0;
                details.tau(j)=tau;
            else
                ops_step_size.separ_vars.y=details_env.Hx-details_env.T.y;
                for i=1:Ns
                    ops_step_size.separ_vars.yt{i}=details_env.Hx_term{i}-details_env.T.yt{i};
                end
                [details.tau(j),details_LS]=obj.LS_backtrackingVersion2(Grad_env,...
                    Zint,W,dir_env,ops_step_size);
                
                details.inner_loops(j,1)=details_LS.inner_loops;
                details.phi_diff(j,1)=details_LS.phi_diff;
                
                tau=details.tau(j);
            end
            

            Wnew.y=W.y+tau*dir_env.y;
            for i=1:Ns
                Wnew.yt{i,1}=W.yt{i}+tau*dir_env.yt{i};
            end
            
            if(details_LS.term_LS)
                grad_steps=grad_steps+1;
            end
            
            Z=obj.Solve_step(Wnew,x0);
            for i=1:non_leaf
                % Hx
                details_prox.Hx(:,i)=sys.F{i}*Z.X(:,i)+sys.G{i}*Z.U(:,i);
                T.y(:,i)=min(Wnew.y(:,i)/lambda+details_prox.Hx(:,i),sys.g{i});
                Y.y(:,i)=Wnew.y(:,i)+lambda*(details_prox.Hx(:,i)-T.y(:,i));
            end
            
            for i=1:Ns
                % Hx_term
                details_prox.Hx_term{i,1}=sys.Ft{i,1}*Z.X(:,tree.leaves(i));
                T.yt{i,1}=min(Wnew.yt{i}/lambda+details_prox.Hx_term{i,1},sys.gt{i});
                Y.yt{i,1}=Wnew.yt{i}+lambda*(details_prox.Hx_term{i,1}-T.yt{i,1});
            end
            
            Z1=obj.Solve_step(Y,x0);
            
            
            delta_grad=zeros(non_leaf*ny+2*Ns*sys.nx,1);
            delta_y=zeros(non_leaf*ny+2*Ns*sys.nx,1);
            
            delta_y(1:ny*non_leaf,1)=vec(Y.y-Wnew.y);
            delta_y(ny*non_leaf+1:end,1)=vec(cell2mat(Y.yt)-cell2mat(Wnew.yt));
            
            for i=1:non_leaf
                % Hx
                delta_grad((i-1)*ny+1:i*ny,1)=sys.F{i}*Z1.X(:,i)+...
                    sys.G{i}*Z1.U(:,i)-details_prox.Hx(:,i);
            end
            
            for i=1:Ns
                delta_grad(ny*non_leaf+2*(i-1)*sys.nx+1:ny*non_leaf+...
                    2*i*sys.nx,1)=sys.Ft{i,1}*Z1.X(:,tree.leaves(i))-details_prox.Hx_term{i,1};
            end
            
            phi_diff=0;
            for i=1:non_leaf
                phi_diff=phi_diff+tree.prob(i)*(Z.X(:,i)'*V.Q*Z.X(:,i)+Z.U(:,i)'*V.R*Z.U(:,i))-...
                    tree.prob(i)*(Z1.X(:,i)'*V.Q*Z1.X(:,i)+Z1.U(:,i)'*V.R*Z1.U(:,i));
            end
            
            for i=1:Ns
                phi_diff=phi_diff+tree.prob(non_leaf+i)*Z.X(:,non_leaf+i)'*V.Vf{i}*...
                    Z.X(:,non_leaf+i)-tree.prob(non_leaf+i)*Z1.X(:,non_leaf+i)'*V.Vf{i}*...
                    Z1.X(:,non_leaf+i);
            end
            
            %phi_diff1=-phi_c-phi_c_dir+phi0+phi0_dir+[vec(details.Hx);cell2mat(details.Hx_term)]'*delta_y...
            %    -0.5*norm(delta_y)^2/lambda;
            
            phi_diff=phi_diff-[vec(Y.y);vec(cell2mat(Y.yt))]'*delta_grad-...
                0.5*norm(delta_y)^2/lambda;
            
            %lambda*norm(delta_grad)>alpha*norm(delta_y)
            %if(lambda*norm(delta_grad)>alpha*norm(delta_y))
            if(phi_diff>0)
                kk=kk+1;
                lambda=beta*lambda;
                obj.algo_details.ops_FBE.lambda=lambda;
                %obj.algo_details.ops_FBE.Lbfgs.LBFGS_col=Lbfgs_col;
                obj.algo_details.ops_FBE.Lbfgs=MLBFGS;
            else
                break;
            end 
        end
        Wold=W;
        Grad_envOld=Grad_env;
        details_prox.T=T;
        %obj.algo_details.ops_FBE.lambda=lambda;
        details.lambda_prox(1,j)=lambda;
        details.inner_prox(1,j)=kk;
        %memory=obj.algo_details.ops_FBE.memory;
    else
       % step 3 primal, 
       Z=obj.Solve_step(W,x0);
       
       %step 4 gradient projection algorithm
       [Y,details_prox]=obj.GobalFBS_proximal_gcong(Z,W);
       details.lambda_prox(1,j)=details_prox.lambda; 
       obj.algo_details.ops_FBE.lambda=details_prox.lambda;
       
       % calculation of the gradient of the envelope
       Grad_env.y=-(details_prox.Hx-details_prox.T.y);
       for i=1:Ns
           Grad_env.yt{i,1}=-(details_prox.Hx_term{i,1}-details_prox.T.yt{i,1});
       end 
       
       Zdir=obj.Solve_step_direction(Grad_env);
       
       for i=1:non_leaf
           Hd.y(:,i)=-(sys.F{i}*Zdir.X(:,i)+sys.G{i}*Zdir.U(:,i));
       end
       
       for i=1:Ns
           Hd.yt{i,1}=-sys.Ft{i}*Zdir.X(:,non_leaf+i);
       end
       
       Grad_env.y=Grad_env.y-details_prox.lambda*Hd.y;
       for i=1:Ns
           Grad_env.yt{i,1}=Grad_env.yt{i,1}-details_prox.lambda*Hd.yt{i,1};
       end
       
       Grad_envOld=Grad_env;
       Wold=W;
       Lbfgs_loop=1;
        
    end
    
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
    
    %details.prm_infs(1,j)=max(max(abs(prm_infs.y)));
    Glambda=[vec(prm_infs.y);vec(cell2mat(prm_infs.yt))];
    details.Glambda(j)=norm(Glambda);
    %max(max(ops_step_size.separ_vars.y))
    if(norm(prm_infs.y)<ops.primal_inf)
        details.iter=j;
        obj.algo_details.ops_FBE.Lbfgs;
        break
    else
        %theta(1)=theta(2);
        %theta(2)=(sqrt(theta(1)^4+4*theta(1)^2)-theta(1)^2)/2;
        j=j+1;
    end
    
end
%%
%details.lambda_prox(1,j)=obj.algo_details.ops_FBE.lambda;

details.dual_gap=0;
for i=1:non_leaf
    details.dual_gap=details.dual_gap+Y.y(:,i)'*prm_infs.y(:,i);
end

for i=1:Ns
    details.dual_gap=details.dual_gap+Y.yt{i}'*prm_infs.yt{i};
end

%obj.algo_details.ops_FBE.Lbfg
details.iter=j;
%details.prm_infs=prm_infs;
details.Hx=details_prox.Hx;
details.T=details_prox.T;
details.grad_steps=grad_steps;
details.FBE_solve=toc;
details.W=W;


end




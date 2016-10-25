function [Z,Y,details]=Dual_FBEAdaptive(obj,x0)
% This function calculate the optimal solution using the
% LBFGS algoirthm on the dual FBE envelpe algorithm at the
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
nx=sys.nx;
nu=sys.nu;
nz=(nx+nu);
ny=2*nz; 
beta=0.5;


% options for the line search
ops_step_size.x0=x0;
ops_step_size.iter_max=50;
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
%dual_grad=prm_fes;
%dual_grad_term=prm_fes_term;

while(j<ops.steps)
    % Step 1: accelerated step
    W=Y; 
    % step 2 : evaluation of the gradient of envelop.
    if(j==1)
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
        
        prm_infs=Grad_env;
        
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
        
    else
        kk=0;
        %lambda_bef=obj.algo_details.ops_FBE.lambda;
        while(kk<5)
            if(strcmp(ops.direction,'LBFGS'))
                MLBFGS=obj.algo_details.ops_FBE.Lbfgs;
                [Grad_env,Z,details_env] =obj.grad_dual_envelopVersion2(W,x0);
                lambda=obj.algo_details.ops_FBE.lambda;
                details.lambda_prox(j)=details_env.lambda;
                
                % calculate the direction by LBFGS method
                [obj,dir_env]= obj.LBFGS_direction(Grad_env,Grad_envOld,W,Wold);
                details.H(j)=obj.algo_details.ops_FBE.Lbfgs.H;
            
            else
                %obj.algo_details.ops_FBE.direction
                [Grad_env,Z,details_env] =obj.grad_dual_envelopVersion2(W,x0);
                lambda=obj.algo_details.ops_FBE.lambda;
                details.lambda_prox(j)=details_env.lambda;
                
                % calcualte the direction using FR-CG method
                
                [dir_env,details.CG_beta(j)]=obj.CG_direction(Grad_env);
            end 
            
            
            
            details.direction(j)=0;
            for i=1:non_leaf
                details.direction(j)=details.direction(j)+dir_env.y(:,i)'*Grad_env.y(:,i);
            end
            for i=1:Ns
                details.direction(j)=details.direction(j)+dir_env.yt{i}'*Grad_env.yt{i};
            end
            %if(abs(details.direction(j))<1e-3)
             %   details.inner_loops(j,1)=0;
             %   details.tau(j)=0;
            %else
                ops_step_size.separ_vars.y=details_env.Hx-details_env.T.y;
                for i=1:Ns
                    ops_step_size.separ_vars.yt{i}=details_env.Hx_term{i}-details_env.T.yt{i};
                end
                if(strcmp(obj.algo_details.ops_FBE.LS,'WOLFE'))
                    [details.tau(j),details_LS]=obj.WolfCndVersion3(Grad_env,Z,W,dir_env,ops_step_size);
                    
                    if(details.tau(j)==0)
                        [details.tau(j) j]
                    end 
                    %details.inner_loops(j,1)=details_LS.inner_iter;
                    
                else
                    ops_step_size.Hx=details_prox.Hx;
                    ops_step_size.Hx_term=details_prox.Hx_term;
                    ops_step_size.T=details_prox.T;
                    [details.lambda(j),details_LS]=obj.Goldstein_conditions...
                        (Grad_env,Z,W,dir_env,ops_step_size);
                    details.inner_loops(j,1)=details_LS.inner_loops;
                end
                
                %details.inner_loops(j,1)=details_LS.inner_loops;
                %details.phi_diff(j,1)=details_LS.phi_diff;
                %end
            if(details_LS.term_LS)
                details_LS.term_LS
                details.tau(j)=0;
            %else
                
            end
            tau=details.tau(j);
            Y.y=W.y+tau*dir_env.y;
            for i=1:Ns
                Y.yt{i}=W.yt{i}+tau*dir_env.yt{i};
            end
            
            Z=obj.Solve_step(Y,x0);
            for i=1:non_leaf
                % Hx
                details_prox.Hx(:,i)=sys.F{i}*Z.X(:,i)+sys.G{i}*Z.U(:,i);
                T.y(:,i)=min(Y.y(:,i)/lambda+details_prox.Hx(:,i),sys.g{i});
                Ynew.y(:,i)=Y.y(:,i)+lambda*(details_prox.Hx(:,i)-T.y(:,i));
            end
            
            for i=1:Ns
                % Hx_term
                details_prox.Hx_term{i,1}=sys.Ft{i,1}*Z.X(:,tree.leaves(i));
                T.yt{i,1}=min(Y.yt{i}/lambda+details_prox.Hx_term{i,1},sys.gt{i});
                Ynew.yt{i,1}=Y.yt{i}+lambda*(details_prox.Hx_term{i,1}-T.yt{i,1});
            end
            
            Z1=obj.Solve_step(Ynew,x0);
            
            
            delta_grad=zeros(non_leaf*ny+2*Ns*sys.nx,1);
            delta_y=zeros(non_leaf*ny+2*Ns*sys.nx,1);
            
            delta_y(1:ny*non_leaf,1)=vec(Ynew.y-Y.y);
            delta_y(ny*non_leaf+1:end,1)=vec(cell2mat(Ynew.yt)-cell2mat(Y.yt));
            
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
            
            phi_diff=phi_diff-[vec(Ynew.y);vec(cell2mat(Ynew.yt))]'*delta_grad-...
                0.5*norm(delta_y)^2/lambda;
            
            %lambda*norm(delta_grad)>alpha*norm(delta_y)
            %if(lambda*norm(delta_grad)>alpha*norm(delta_y))
            if(phi_diff>0 || details_LS.term_LS)
                kk=kk+1;
                lambda=beta*lambda;
                obj.algo_details.ops_FBE.lambda=lambda;
                %ops_step_size.iter_max=2*ops_step_size.iter_max;
                %obj.algo_details.ops_FBE.Lbfgs.LBFGS_col=Lbfgs_col;
                if(strcmp(obj.algo_details.ops_FBE.direction,'LBFGS'))
                    obj.algo_details.ops_FBE.Lbfgs=MLBFGS;
                end 
            else
                break;
            end 
            %{
            if(kk==5)
                % if you cannot find the tau or lamda
                % use a simple proximal step.
                obj.algo_details.ops_FBE.lambda=lambda_bef;
                [Y,details_prox]=obj.GobalFBS_proximal_gcong(Z,W);
                details.lambda_prox(1,j)=details_prox.lambda;
                obj.algo_details.ops_FBE.lambda=details_prox.lambda;
                
                % calculation of the gradient of the envelope
                Grad_env.y=-(details_prox.Hx-details_prox.T.y);
                for i=1:Ns
                    Grad_env.yt{i,1}=-(details_prox.Hx_term{i,1}-details_prox.T.yt{i,1});
                end
                
                prm_infs=Grad_env;
                
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
                dir_env=Grad_env;
            end
            %}
        end
        prm_infs.y=details_env.Hx-details_env.T.y;
        for i=1:Ns
            prm_infs.yt{i}=details_env.Hx_term{i}-details_env.T.yt{i};
        end
    end
    %% 
    
    if(strcmp(obj.algo_details.ops_FBE.direction,'LBFGS'))
        % storing the past gradient for LBFGS method 
        Grad_envOld=Grad_env;
        Wold=W;
    else 
        obj.algo_details.ops_FBE.ConjGrad...
            .prev_grad_norm(1,j)=norm([vec(Grad_env.y);cell2mat(Grad_env.yt)],2)^2;
        
        obj.algo_details.ops_FBE.ConjGrad.iterate=j;
        
        if(j==1)
            obj.algo_details.ops_FBE.ConjGrad.prev_dir=Grad_env; 
        else
            obj.algo_details.ops_FBE.ConjGrad.prev_dir=dir_env;
        end
        
        details.CG_prev_grad_norm(j)=norm([vec(Grad_env.y);cell2mat(Grad_env.yt)],2);
        
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
    
    % Glambda
    Glambda=[vec(prm_infs.y);vec(cell2mat(prm_infs.yt))];
    details.Glambda(j)=norm(Glambda);
    
    % [norm(prm_infs.y) j]
    % norm(prm_infs.y,2);
    if(norm(prm_infs.y)<ops.primal_inf)
        details.iter=j;
        %obj.algo_details.ops_FBE.Lbfgs;
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
    details.dual_gap=details.dual_gap+Y.y(:,i)'*prm_infs.y(:,i);
end

for i=1:Ns
    details.dual_gap=details.dual_gap+Y.yt{i}'*prm_infs.yt{i};
end


%obj.algo_details.ops_FBE.Lbfg
details.iter=j;
details.separ_var.y=prm_infs.y;
details.separ_var.yt=prm_infs.yt;
details.Hx=details_prox.Hx;
details.T=details_prox.T;
details.FBE_solve=toc;
details.W=W;



end
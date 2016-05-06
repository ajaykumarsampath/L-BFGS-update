function [ Grad_env,Z,details] = grad_dual_envelop( obj,Y,x0)
%
% This function calcualte the gradient of the envelop 
% function 
% 
% Syntax: 
% INPUT:     Y        :  Dual variable 
% 
% OUTPUT:    Grad_env :  Gradient of the dual variable 
%            W        :  Updated dual variable
%            details  :  Structure containing the 
%

sys=obj.SysMat_.sys;
tree=obj.SysMat_.tree;

%lambda=0.1;
Ns=length(tree.leaves);
Nd=length(tree.stage);
non_leaf=Nd-Ns;

% calculation of the dual gradient; 
Z=obj.Solve_step(Y,x0);
details.T.y=zeros(2*(sys.nx+sys.nu),non_leaf);
details.Hx=zeros(2*(sys.nx+sys.nu),non_leaf);

% calculation of the proximal with g 

if(strcmp(obj.algo_details.ops_FBE.prox_LS,'yes'))
    % Backtracking is used to calcualte the step-size
    lambda=obj.algo_details.ops_FBE.lambda;
    nz=sys.nx+sys.nu;
    ny=2*nz;
    alpha=obj.algo_details.ops_FBE.alphaB;
    beta=obj.algo_details.ops_FBE.betaB;
    delta_grad=zeros(non_leaf*ny+2*Ns*sys.nx,1);
    delta_y=zeros(non_leaf*ny+2*Ns*sys.nx,1);
    
    for i=1:non_leaf
        % Hx
        details.Hx(:,i)=sys.F{i}*Z.X(:,i)+sys.G{i}*Z.U(:,i); 
    end 
    
    for i=1:Ns
        details.Hx_term{i,1}=sys.Ft{i,1}*Z.X(:,tree.leaves(i));
    end
    
    while(1)
        
        % calculateing the prox with respect to g conjugate 
        for i=1:non_leaf
            T.y(:,i)=min(lambda*Y.y(:,i)+details.Hx(:,i),sys.g{i});
            Y1.y(:,i)=Y.y(:,i)+lambda*(details.Hx(:,i)-T.y(:,i));
        end 
        
        for i=1:Ns
            T.yt{i,1}=min(lambda*Y.yt{i,1}+details.Hx_term{i,1},sys.gt{i});
            Y1.yt{i,1}=Y.yt{i,1}+lambda*(details.Hx_term{i,1}-T.yt{i,1});
        end
        
        delta_y(1:ny*non_leaf,1)=vec(Y1.y-Y.y);
        delta_y(ny*non_leaf+1:end,1)=vec(cell2mat(Y1.yt)-cell2mat(Y.yt));
        
        Z1=obj.Solve_step(Y1,x0);
        
        for i=1:non_leaf
            % Hx
            delta_grad((i-1)*ny+1:i*ny,1)=sys.F{i}*Z1.X(:,i)+...
                sys.G{i}*Z1.U(:,i)-details.Hx(:,i);
        end
        
        for i=1:Ns
            delta_grad(ny*non_leaf+2*(i-1)*sys.nx+1:ny*non_leaf+...
                2*i*sys.nx,1)=sys.Ft{i,1}*Z.X(:,tree.leaves(i))-details.Hx_term{i,1};
        end
        
        %lambda*norm(delta_grad)>alpha*norm(delta_y)
        if(lambda*norm(delta_grad)>alpha*norm(delta_y))
            lambda=beta*lambda;
        else
            break;
        end
    end 
    details.T=T;
else 
    % without backtracking
    lambda=obj.algo_details.ops_FBE.lambda;
    
    for i=1:non_leaf
        % Hx
        details.Hx(:,i)=sys.F{i}*Z.X(:,i)+sys.G{i}*Z.U(:,i);
        details.T.y(:,i)=min(lambda*Y.y(:,i)+details.Hx(:,i),sys.g{i});
    end
    
    for i=1:Ns
        details.Hx_term{i,1}=sys.Ft{i,1}*Z.X(:,tree.leaves(i));
        details.T.yt{i,1}=min(lambda*Y.yt{i,1}+details.Hx_term{i,1},sys.gt{i});
    end
    
end
% calculation of the proximal_gconj update 
%[W,details_prox_gconj]=obj.proximal_gconj(Z,Y);

% y-prox_{g^\star}(y-\gamma\Delta f^{\star}(-H'y))
%Grad_env.y=details.Hx-details.T.y; 
Grad_env.y=-(details.Hx-details.T.y);
for i=1:Ns
    %Grad_env.yt{i,1}=(details.Hx_term{i,1}-details.T.yt{i,1};
    Grad_env.yt{i,1}=-(details.Hx_term{i,1}-details.T.yt{i,1});
end

Grad_env2=Grad_env;
% Hessian-free evaluation: 
%Hd1=obj.dual_hessian_free(Y,Grad_env,Z);
Zdir=obj.Solve_step_direction(Grad_env);

for i=1:non_leaf
    Hd.y(:,i)=-(sys.F{i}*Zdir.X(:,i)+sys.G{i}*Zdir.U(:,i));
end

for i=1:Ns
   Hd.yt{i,1}=-sys.Ft{i}*Zdir.X(:,non_leaf+i);
end

Grad_env.y=Grad_env.y-lambda*Hd.y;
for i=1:Ns
    Grad_env.yt{i,1}=Grad_env.yt{i,1}-lambda*Hd.yt{i,1};
end 

pos_def_test=0;
for i=1:non_leaf
    pos_def_test=pos_def_test+Grad_env.y(:,i)'*Grad_env2.y(:,i);
end 

for i=1:Ns
    pos_def_test=pos_def_test+Grad_env.yt{i,1}'*Grad_env2.yt{i,1};
end 

%pos_def_test
details.pos_def=pos_def_test;
details.lambda=lambda;
end


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
V=obj.SysMat_.V;

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
        % Hx_term
        details.Hx_term{i,1}=sys.Ft{i,1}*Z.X(:,tree.leaves(i));
    end
    
    %{
    phi0=0;
    phi0_dir=0;
    phi10=0;
    for i=1:non_leaf
        % Hx
        details.Hx(:,i)=sys.F{i}*Z.X(:,i)+sys.G{i}*Z.U(:,i); 
        phi0=phi0+tree.prob(i)*(Z.X(:,i)'*V.Q*Z.X(:,i)+Z.U(:,i)'*V.R*Z.U(:,i));
        phi0_dir=phi0_dir+Y.y(:,i)'*details.Hx(:,i);
        phi10=phi10+tree.prob(i)*(Z.X(:,i)'*V.Q*Z.X(:,i)+Z.U(:,i)'*V.R*Z.U(:,i))...
            +Y.y(:,i)'*details.Hx(:,i);
    end 
    
    for i=1:Ns
        details.Hx_term{i,1}=sys.Ft{i,1}*Z.X(:,tree.leaves(i));
         phi0=phi0+tree.prob(non_leaf+i)*Z.X(:,non_leaf+i)'*V.Vf{i}*Z.X(:,non_leaf+i);
         phi0_dir=phi0_dir+Y.yt{i,1}'*details.Hx_term{i,1};
         phi10=phi10+tree.prob(non_leaf+i)*Z.X(:,non_leaf+i)'*V.Vf{i}*Z.X(:,non_leaf+i)...
             +Y.yt{i,1}'*details.Hx_term{i,1};
    end
    %}
    
    while(1)
        
        % calculateing the prox with respect to g conjugate 
        for i=1:non_leaf
            T.y(:,i)=min(Y.y(:,i)/lambda+details.Hx(:,i),sys.g{i});
            Y1.y(:,i)=Y.y(:,i)+lambda*(details.Hx(:,i)-T.y(:,i));
        end 
        
        for i=1:Ns
            T.yt{i,1}=min(Y.yt{i,1}/lambda+details.Hx_term{i,1},sys.gt{i});
            Y1.yt{i,1}=Y.yt{i,1}+lambda*(details.Hx_term{i,1}-T.yt{i,1});
        end
        
        %min_y=min(min(min(Y1.y)),min(min(cell2mat(Y1.yt))))
        
        delta_y(1:ny*non_leaf,1)=vec(Y1.y-Y.y);
        delta_y(ny*non_leaf+1:end,1)=vec(cell2mat(Y1.yt)-cell2mat(Y.yt));
        
        
        Z1=obj.Solve_step(Y1,x0);
        %{
        phi_c=0;
        phi_c_dir=0;
        phi_10c=0;
        for i=1:non_leaf
            % Hx
            Hx_c(:,i)=sys.F{i}*Z1.X(:,i)+sys.G{i}*Z1.U(:,i);
            phi_c=phi_c+tree.prob(i)*(Z1.X(:,i)'*V.Q*Z1.X(:,i)+Z1.U(:,i)'*V.R*Z1.U(:,i));
            phi_c_dir=phi_c_dir+Y1.y(:,i)'*Hx_c(:,i);
            phi_10c=phi_10c+tree.prob(i)*(Z1.X(:,i)'*V.Q*Z1.X(:,i)+Z1.U(:,i)'*V.R*Z1.U(:,i))...
                +Y1.y(:,i)'*Hx_c(:,i);
        end
        
        for i=1:Ns
            Hx_term_c{i,1}=sys.Ft{i,1}*Z1.X(:,tree.leaves(i));
            phi_c=phi_c+tree.prob(non_leaf+i)*Z1.X(:,non_leaf+i)'*V.Vf{i}*Z1.X(:,non_leaf+i);
            phi_c_dir=phi_c_dir+Y1.yt{i,1}'*Hx_term_c{i,1};
            phi_10c=phi_10c+tree.prob(non_leaf+i)*Z1.X(:,non_leaf+i)'*V.Vf{i}*Z1.X(:,non_leaf+i)...
                +Y1.yt{i,1}'*Hx_term_c{i,1};
        end
        %}
        for i=1:non_leaf
            % Hx
            delta_grad((i-1)*ny+1:i*ny,1)=sys.F{i}*Z1.X(:,i)+...
                sys.G{i}*Z1.U(:,i)-details.Hx(:,i);
        end
        
        for i=1:Ns
            delta_grad(ny*non_leaf+2*(i-1)*sys.nx+1:ny*non_leaf+...
                2*i*sys.nx,1)=sys.Ft{i,1}*Z1.X(:,tree.leaves(i))-details.Hx_term{i,1};
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
        
        phi_diff=phi_diff-[vec(Y1.y);vec(cell2mat(Y1.yt))]'*delta_grad-0.5*norm(delta_y)^2/lambda;
        
        
        %lambda*norm(delta_grad)>alpha*norm(delta_y)
        %if(lambda*norm(delta_grad)>alpha*norm(delta_y))
        if(phi_diff>0)
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
        details.T.y(:,i)=min(Y.y(:,i)/lambda+details.Hx(:,i),sys.g{i});
    end
    
    for i=1:Ns
        details.Hx_term{i,1}=sys.Ft{i,1}*Z.X(:,tree.leaves(i));
        details.T.yt{i,1}=min(Y.yt{i,1}/lambda+details.Hx_term{i,1},sys.gt{i});
    end
    
end
% calculation of the proximal_gconj update 
%[W,details_prox_gconj]=obj.proximal_gconj(Z,Y);

% y-prox_{g^\star}(y-\gamma\Delta f^{\star}(-H'y))
% Grad_env.y=details.Hx-details.T.y; 
Grad_env.y=-(details.Hx-details.T.y);
for i=1:Ns
    %Grad_env.yt{i,1}=(details.Hx_term{i,1}-details.T.yt{i,1};
    Grad_env.yt{i,1}=-(details.Hx_term{i,1}-details.T.yt{i,1});
end

% value function on the evelope 
phi=0;
for j=1:non_leaf
    phi=phi+tree.prob(j)*Z.X(:,j)'*V.Q*Z.X(:,j)+tree.prob(j)*Z.U(:,j)'*V.R*Z.U(:,j)+...
        0.5*lambda*norm(Grad_env.y(:,j))^2-Y.y(:,j)'*Grad_env.y(:,j);
end

for j=1:Ns
    phi=phi+tree.prob(non_leaf+j)*Z.X(:,non_leaf+j)'*V.Vf{j}*Z.X(:,non_leaf+j)-...
        Y.yt{j,1}'*Grad_env.yt{j,1}+0.5*lambda*norm(Grad_env.yt{j,1})^2;
end
details.phi=-phi;

Grad_env2=Grad_env;
% Hessian-free evaluation: 
% Hd1=obj.dual_hessian_free(Y,Grad_env,Z);
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

%
pos_def_test=0;
for i=1:non_leaf
    pos_def_test=pos_def_test+Grad_env.y(:,i)'*Grad_env2.y(:,i);
end 

for i=1:Ns
    pos_def_test=pos_def_test+Grad_env.yt{i,1}'*Grad_env2.yt{i,1};
end
details.pos_def=pos_def_test;
%}
details.lambda=lambda;
%details.Y1=Y1;

end


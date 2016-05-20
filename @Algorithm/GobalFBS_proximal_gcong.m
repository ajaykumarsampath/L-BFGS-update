function [Y,details_prox]=GobalFBS_proximal_gcong(obj,Z,W)
%  This function calculate the proximal of the hard constraints and
%  update the dual variables
%
%  Syntax   [Y,details_prox]=proximal_gconj(Z,W)
%
%  INPUT:
%                     Z   :  Z.X, Z.U
%                     W   :  dual variables
%
%  OUTPUT             Y   :  dual varible
%          details_prox   :  proximal varibles
%

sys=obj.SysMat_.sys;
tree=obj.SysMat_.tree;

%lambda=0.1;
Ns=length(tree.leaves);
Nd=length(tree.stage);
non_leaf=Nd-Ns;

x0=Z.X(:,1);
% Hx,z 
details_prox.T.y=zeros(2*(sys.nx+sys.nu),non_leaf);
details_prox.Hx=zeros(2*(sys.nx+sys.nu),non_leaf);

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
        details_prox.Hx(:,i)=sys.F{i}*Z.X(:,i)+sys.G{i}*Z.U(:,i); 
    end 
    
    for i=1:Ns
        details_prox.Hx_term{i,1}=sys.Ft{i,1}*Z.X(:,tree.leaves(i));
    end
    
    while(1)
        
        % calculateing the prox with respect to g conjugate 
        for i=1:non_leaf
            T.y(:,i)=min(W.y(:,i)/lambda+details_prox.Hx(:,i),sys.g{i});
            Y.y(:,i)=W.y(:,i)+lambda*(details_prox.Hx(:,i)-T.y(:,i));
        end  
        
        for i=1:Ns
            T.yt{i,1}=min(W.yt{i,1}/lambda+details_prox.Hx_term{i,1},sys.gt{i});
            Y.yt{i,1}=W.yt{i,1}+lambda*(details_prox.Hx_term{i,1}-T.yt{i,1});
        end
        
        delta_y(1:ny*non_leaf,1)=vec(Y.y-W.y);
        delta_y(ny*non_leaf+1:end,1)=vec(cell2mat(Y.yt)-cell2mat(W.yt));
        
        Z1=obj.Solve_step(Y,x0);
        
        for i=1:non_leaf
            % delta gradient
            delta_grad((i-1)*ny+1:i*ny,1)=sys.F{i}*Z1.X(:,i)+...
                sys.G{i}*Z1.U(:,i)-details_prox.Hx(:,i);
        end
        
        for i=1:Ns
            delta_grad(ny*non_leaf+2*(i-1)*sys.nx+1:ny*non_leaf+...
                2*i*sys.nx,1)=sys.Ft{i,1}*Z.X(:,tree.leaves(i))-details_prox.Hx_term{i,1};
        end
        
        %lambda*norm(delta_grad)>alpha*norm(delta_y)
        if(lambda*norm(delta_grad)>alpha*norm(delta_y))
            lambda=beta*lambda;
        else
            details_prox.lambda=lambda;
            break;
        end
    end 
    details_prox.T=T;
else 
    % without backtracking
    lambda=obj.algo_details.ops_FBE.lambda;
    
    for i=1:non_leaf
        % Hx
        details_prox.Hx(:,i)=sys.F{i}*Z.X(:,i)+sys.G{i}*Z.U(:,i);
        details_prox.T.y(:,i)=min(W.y(:,i)/lambda+details_prox.Hx(:,i),sys.g{i});
        Y.y(:,i)=W.y(:,i)+lambda*(details_prox.Hx(:,i)-details_prox.T.y(:,i));
    end
    
    for i=1:Ns
        details_prox.Hx_term{i,1}=sys.Ft{i,1}*Z.X(:,tree.leaves(i));
        details_prox.T.yt{i,1}=min(W.yt{i,1}/lambda+details_prox.Hx_term{i,1},sys.gt{i});
        Y.yt{i,1}=W.yt{i,1}+lambda*(details_prox.Hx_term{i,1}-details_prox.T.yt{i,1});
    end
    
end




end


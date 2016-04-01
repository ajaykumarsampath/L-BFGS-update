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
lambda=obj.algo_details.ops_FBS.lambda;
Ns=length(tree.leaves);
Nd=length(tree.stage);
non_leaf=Nd-Ns;

% calculation of the dual gradient; 
Z=obj.Solve_step(Y,x0);
details.T.y=zeros(2*(sys.nx+sys.nu),non_leaf);
details.Hx=zeros(2*(sys.nx+sys.nu),non_leaf);
% calculation of the proximal with g 
for i=1:non_leaf
    % Hx
    details.Hx(:,i)=sys.F{i}*Z.X(:,i)+sys.G{i}*Z.U(:,i);
    details.T.y(:,i)=min(lambda*Y.y(:,i)+details.Hx(:,i),sys.g{i});
    %details_prox.dual_grad(:,i)=details_prox.prm_fes(:,i)-sys.g{i};
end

for i=1:Ns
    details.Hx_term{i,1}=sys.Ft{i,1}*Z.X(:,tree.leaves(i));
    details.T.yt{i,1}=min(lambda*Y.yt{i,1}+details.Hx_term{i,1},sys.gt{i});
    %W.yt{i,1}=Y.yt{i,:}+alpha*(details.Hx_term{i,1}-details.T.yt{i,1});
end

% calculation of the proximal_gconj update 
%[W,details_prox_gconj]=obj.proximal_gconj(Z,Y);

% y-prox_{g^\star}(y-\gamma\Delta f^{\star}(-H'y))
Grad_env.y=details.Hx-details.T.y; 
for i=1:Ns
    Grad_env.yt{i,1}=details.Hx_term{i,1}-details.T.yt{i,1};
end


% Hessian-free evaluation: 
Hd=obj.dual_hessian_free(Y,Grad_env,Z);

Grad_env.y=Grad_env.y-Hd.y;
for i=1:Ns
    Grad_env.yt{i,1}=Grad_env.yt{i,1}-Hd.yt{i,1};
end 

end


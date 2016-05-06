function [Y,details_prox]=proximal_gconj(obj,Z,W)
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
if(strcmp(obj.algo_details.ops_APG.prox_LS,'yes'))
    
    %{
    dual_grad=zeros(2*(sys.nx+sys.nu),tree.ancestor(tree.leaves(end)));
    prm_fes=zeros(2*(sys.nx+sys.nu),tree.ancestor(tree.leaves(end)));
    
    dual_grad_term=cell(length(tree.leaves),1);
    prm_fes_term=cell(length(tree.leaves),1);
    %}
    prev_lambda=obj.algo_details.ops_APG.lambda;
    
    for i=1:tree.ancestor(tree.leaves(end))
        details_prox.dual_grad(:,i)=sys.F{i}*Z.X(:,i)+sys.G{i}*Z.U(:,i);
        details_prox.prm_fes(:,i)=details_prox.dual_grad(:,i)-sys.g{i};
    end
    
    for i=1:length(tree.leaves)
        details_prox.dual_grad_term{i,1}=sys.Ft{i,1}*Z.X(:,tree.leaves(i));
        details_prox.prm_fes_term{i,1}=details_prox.dual_grad_term{i,1}-sys.gt{i};
    end
    
    ops_backtrack.x0=Z.X(:,1);
    ops_backtrack.W=W;
    
    ops_backtrack.dual_grad.y=details_prox.dual_grad;
    ops_backtrack.dual_grad.yt=details_prox.dual_grad_term;
    
    ops_backtrack.prm_fes.y=details_prox.prm_fes;
    ops_backtrack.prm_fes.yt=details_prox.prm_fes_term;

    lambda=obj.backtacking_prox_method( prev_lambda,ops_backtrack);
    
    for i=1:tree.ancestor(tree.leaves(end))
        Y.y(:,i)=max(0,W.y(:,i)+lambda*(details_prox.prm_fes(:,i)));
    end
    
    for i=1:length(tree.leaves)
        Y.yt{i,1}=max(0,W.yt{i,:}+lambda*(details_prox.prm_fes_term{i,1}));
    end
    details_prox.lambda=lambda;
else
    
    lambda=obj.algo_details.ops_APG.lambda;
    details_prox.lambda=lambda;
    
    %lambda=0.15;
    for i=1:tree.ancestor(tree.leaves(end))
        details_prox.dual_grad(:,i)=sys.F{i}*Z.X(:,i)+sys.G{i}*Z.U(:,i);
        details_prox.prm_fes(:,i)=details_prox.dual_grad(:,i)-sys.g{i};
        Y.y(:,i)=max(0,W.y(:,i)+lambda*(details_prox.prm_fes(:,i)));
    end
    
    for i=1:length(tree.leaves)
        details_prox.dual_grad_term{i,1}=sys.Ft{i,1}*Z.X(:,tree.leaves(i));
        details_prox.prm_fes_term{i,1}=details_prox.dual_grad_term{i,1}-sys.gt{i};
        Y.yt{i,1}=max(0,W.yt{i,:}+lambda*(details_prox.prm_fes_term{i,1}));
    end
end




end


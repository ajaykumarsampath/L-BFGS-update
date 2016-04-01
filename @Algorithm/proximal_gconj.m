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
lambda=obj.algo_details.ops_APG.lambda;

for i=1:tree.ancestor(tree.leaves(end))
    details_prox.prm_fes(:,i)=sys.F{i}*Z.X(:,i)+sys.G{i}*Z.U(:,i);
    details_prox.dual_grad(:,i)=details_prox.prm_fes(:,i)-sys.g{i};
    Y.y(:,i)=max(0,W.y(:,i)+lambda*(details_prox.dual_grad(:,i)));
end

for i=1:length(tree.leaves)
    details_prox.prm_fes_term{i,1}=sys.Ft{i,1}*Z.X(:,tree.leaves(i));
    details_prox.dual_grad_term{i,1}=details_prox.prm_fes_term{i,1}-sys.gt{i};
    Y.yt{i,1}=max(0,W.yt{i,:}+lambda*(details_prox.dual_grad_term{i,1}));
end


end


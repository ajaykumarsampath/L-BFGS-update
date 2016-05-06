function [ Hd ] = dual_hessian_free( obj,Y,d,Z)
%
% This function approximate the hessian without calcuating the dual Hessian
% 
% 
%   Hd = [G(y+ed)-Gy]/e
% 
% Syntax :   [ Hd ] = dual_hessian_free( obj,Y,d,x0)
%
% INPUT  :      Y      :   Dual vector    
%               d      :   Updated vector
%               Z      :   state vector, representing the gradient of the
%                          dual vector 
% 
% OUTPUT :      Hd     : approximate of the hessian dual update
%
%

epsilon=obj.algo_details.ops_FBE.epsilon;
sys=obj.SysMat_.sys;
tree=obj.SysMat_.tree;
Nd=length(tree.stage);
non_leaf=length(tree.children);
W.y=Y.y+epsilon*d.y;

for i=1:Nd-non_leaf
    W.yt{i}=Y.yt{i}+epsilon*d.yt{i};
end 


Zepsilon=obj.Solve_step(W,Z.X(:,1));

Hd=Y;

for i=1:non_leaf
    Hd.y(:,i)=-(sys.F{i}*(Zepsilon.X(:,i)-Z.X(:,i))+sys.G{i}*(Zepsilon.U(:,i)-Z.U(:,i)))/epsilon;
end

for i=1:length(tree.leaves)
   Hd.yt{i}=-sys.Ft{i}*(Zepsilon.X(:,non_leaf+i)-Z.X(:,non_leaf+i))/epsilon;
end

end


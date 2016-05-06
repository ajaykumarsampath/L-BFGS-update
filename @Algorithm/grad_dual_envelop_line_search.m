function [ Grad_env,Z,details] = grad_dual_envelop_line_search(obj,Y,D,x0,ops_LS)
%
% This function calcualte the gradient of the envelop 
% function 
% 
% Syntax: [ Grad_env,Z,details] = grad_dual_envelop_line_search( obj,Y,x0,ops_LS)
% 
% INPUT:     Y        :  Dual variable 
%            D        :  Direction of the optimisation
%            x0       :  intial state
% 
% OUTPUT:    Grad_env :  Gradient of the dual variable 
%            W        :  Updated dual variable
%            details  :  Structure containing the 
%



Zcst=ops_LS.Z;
Zdir=ops_LS.Zdir;
alpha=ops_LS.alpha;
Hx=ops_LS.Hx;
HxDir=ops_LS.HxDir;
Hx_term=ops_LS.Hx_term;
Hx_termDir=ops_LS.Hx_term_Dir;

sys=obj.SysMat_.sys;
tree=obj.SysMat_.tree;
lambda=obj.algo_details.ops_FBE.lambda;

Ns=length(tree.leaves);
Nd=length(tree.stage);
non_leaf=Nd-Ns;

% calculation of the dual gradient;
Z.X=Zcst.X+alpha*Zdir.X;
Z.U=Zcst.U+alpha*Zdir.U;

details.T.y=zeros(2*(sys.nx+sys.nu),non_leaf);
% Hx
details.Hx=Hx+alpha*HxDir;
% calculation of the proximal with g 
for i=1:non_leaf  
    details.T.y(:,i)=min(lambda*Y.y(:,i)+details.Hx(:,i),sys.g{i});
end

for i=1:Ns
    details.Hx_term{i,1}=Hx_term{i,1}+alpha*Hx_termDir{i,1};
    details.T.yt{i,1}=min(lambda*Y.yt{i,1}+details.Hx_term{i,1},sys.gt{i});
end


% y-prox_{g^\star}(y-\gamma\Delta f^{\star}(-H'y))
Grad_env.y=-(details.Hx-details.T.y);
for i=1:Ns
    Grad_env.yt{i,1}=-(details.Hx_term{i,1}-details.T.yt{i,1});
end


% Hessian-free evaluation: 
Hd=obj.dual_hessian_free(Y,Grad_env,Z);

Grad_env.y=Grad_env.y-lambda*Hd.y;
for i=1:Ns
    Grad_env.yt{i,1}=Grad_env.yt{i,1}-lambda*Hd.yt{i,1};
end 

end




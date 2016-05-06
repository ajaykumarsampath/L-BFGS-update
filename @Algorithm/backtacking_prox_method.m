function [ lambda ] = backtacking_prox_method(obj,prev_lambda,ops_backtrack)
% This function calculates the step-size for the proximal-gradient 
% algorithm. This is calculated using Backtracking algorithm based the 
% Lipschitz condition. 
% 
% Syntax : [ lambda ] = backtacking_prox_method( prev_lambda)
%
% Input  : prev_lambda  :  Previous step-size 
%          ops_stepsize :  Options of the step-size
%
% Output : lambda       :  Current step-size
%
%

sys=obj.SysMat_.sys;
tree=obj.SysMat_.tree;
Nd=length(tree.stage);
Ns=length(tree.leaves);
non_leaf=Nd-Ns;
ny=2*(sys.nx+sys.nu);

x0=ops_backtrack.x0;
W1=ops_backtrack.W;
dual_grad=ops_backtrack.dual_grad;
prm_fes=ops_backtrack.prm_fes;

dual_grad2.y=zeros(size(dual_grad.y));
%dual_grad2.y=zeros(2*(sys.nx+sys.nu),non_leaf);
dual_grad2.yt=cell(Ns,1);

delta_grad=zeros(ny*non_leaf+2*Ns*sys.nx,1);
delta_y=zeros(ny*non_leaf+2*Ns*sys.nx,1);

lambda=prev_lambda;
beta=0.5;
alpha=0.5;

while(1)
    
    W2.y=max(0,W1.y+lambda*prm_fes.y);
    
    for i=1:Ns
        W2.yt{i,1}=max(0,W1.yt{i,1}+lambda*(prm_fes.yt{i,1}));
    end
    
    Z2=obj.Solve_step(W2,x0);
    
    for i=1:non_leaf
        dual_grad2.y(:,i)=sys.F{i}*Z2.X(:,i)+sys.G{i}*Z2.U(:,i);
    end
    
    for i=1:length(tree.leaves)
        dual_grad2.yt{i,1}=sys.Ft{i,1}*Z2.X(:,tree.leaves(i));
    end
    
    delta_grad(1:ny*non_leaf,1)=vec(dual_grad2.y-dual_grad.y);
    delta_grad(ny*non_leaf+1:ny*non_leaf+2*Ns*sys.nx,1)=vec(cell2mat(...
        dual_grad2.yt)-cell2mat(dual_grad.yt));
    
    delta_y(1:ny*non_leaf,1)=vec(W2.y-W1.y);
    delta_y(ny*non_leaf+1:ny*non_leaf+2*Ns*sys.nx,1)=vec(cell2mat(...
        W2.yt)-cell2mat(W1.yt));
    
    %lambda*norm(delta_grad)>alpha*norm(delta_y)
    
    if(lambda*norm(delta_grad)>alpha*norm(delta_y))
        lambda=beta*lambda;
    else
        break
    end
    
end
%{
 prm_fes(:,i)=dual_grad2.y(:,i)-sys.g{i};
 Y2.y(:,i)=max(0,Y1.y(:,i)+lambda*(prm_fes(:,i)));
 
 prm_fes{i,1}=dual_grad2.yt{i,1}-sys.gt{i};
 Y2.yt{i,1}=max(0,Y1.yt{i,:}+lambda*(prm_fes{i,1}));
 %}
end


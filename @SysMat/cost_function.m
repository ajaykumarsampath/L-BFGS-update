function [ value,primal_epsilon ] = cost_function( obj,Z)
%
% This function calcualtes the primal infeasibility and the primal 
% objective
% 
% SYNTAX : [ value,primal_epsilon ] = cost_funciton( obj,Z)
% 
% INPUT  :          obj    :  system dynamics 
%                   Z      :  optimal state and control 
%
% OUTPUT :          value  :  primal value 
%                   primal :  primal epsilon 
%


sys=obj.sys;
V=obj.V;
tree=obj.tree;

Nd=length(tree.stage);
Ns=length(tree.leaves);
non_leaf=Nd-Ns;

value=0;
epsilon=zeros(Nd,1);

for i=1:non_leaf
    value=value+tree.prob(i)*(Z.X(:,i)'*V.Q*Z.X(:,i)+Z.U(:,i)'*V.R*Z.U(:,i));
    
    epsilon(i,1)=max(max(sys.F{i}*Z.X(:,i)+sys.G{i}*Z.U(:,i)-sys.g{i},0));
end 

for i=1:Ns
    j=tree.leaves(i);
    value=value+tree.prob(j)*(Z.X(:,j)'*V.Vf{i}*Z.X(:,j));
    
    epsilon(j,1)=max(max(sys.Ft{i}*Z.X(:,j)-sys.gt{i},0));
end

primal_epsilon=max(epsilon);

end




function [ sys ] = scale_constraints_system( SysMat)
%
% This function normalises the constraints in the system
%
% Syntax : sys=scale_constraints_system(sys)
%
% INPUT : 
%
%

sys=SysMat.sys;
tree=SysMat.tree;
Nd=length(tree.stage);
Ns=length(tree.leaves);

% Normalising the constraints at the non-leaf nodes.
nc=size(sys.F{1},1);
for i=1:Nd-Ns
    for j=1:nc
        if(abs(sys.g{i}(j))>0)
            sys.F{i}(j,:)=sys.F{i}(j,:)/abs(sys.g{i}(j));
            sys.G{i}(j,:)=sys.G{i}(j,:)/abs(sys.g{i}(j));
            sys.g{i}(j)=sys.g{i}(j)/abs(sys.g{i}(j));
        end
    end
end

% Normalising the terminal constraints
nc=size(sys.Ft{1},1);
for i=1:Ns
    for j=1:nc
        if(abs(sys.gt{i}(j))>0)
            sys.Ft{i}(j,:)=sys.Ft{i}(j,:)/sys.gt{i}(j);
            sys.gt{i}(j)=sys.gt{i}(j)/abs(sys.gt{i}(j));
        end
    end
end

end


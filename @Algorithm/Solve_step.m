function [Z,Q]=Solve_step(obj,Y,xinit)
% This function calculate the on-line computaiton of the dual gradient on
% the tree. 
% 
% Syntax : [Z,Q]=Solve_step(obj,Y,xinit);
% 
% INPUT:         Y     :    dual variable
%              xinit   :    Initial state
% 
% Z is the output and containing


sys=obj.SysMat_.sys;
Ptree=obj.Ptree_;
Tree=obj.SysMat_.tree;

Z.X=zeros(sys.nx,Tree.leaves(end));
Z.U=zeros(sys.nu,Tree.ancestor(Tree.leaves(end)));
S=zeros(sys.nu,Tree.ancestor(Tree.leaves(end)));

q=zeros(sys.nx,Tree.ancestor(Tree.leaves(end)));
qt=cell(1,length(Tree.leaves));
for i=1:length(Tree.leaves)
    %q(:,Tree.leaves(i))=sys.Ft{i,1}'*Y.yt{i,:}';
    qt{1,i}=Y.yt{i,:}';
end

for i=sys.Np:-1:1
    nodes_stage=find(Tree.stage==i-1);
    total_nodes=length(nodes_stage);
    for j=1:total_nodes
        no_child=length(Tree.children{nodes_stage(j)});
        if(no_child>1)
            sum_u=zeros(sys.nu,1);
            sum_q=zeros(sys.nx,1);
            for k=1:no_child
                %Tree.children{nodes_stage(j)}(k)
                sum_u=sum_u+Ptree.Theta{Tree.children{nodes_stage(j)}(k)-1}*q(:,Tree.children{nodes_stage(j)}(k));
                sum_q=sum_q+Ptree.f{Tree.children{nodes_stage(j)}(k)-1}'*q(:,Tree.children{nodes_stage(j)}(k));
            end
            Z.sum_u{nodes_stage(j)}=sum_u;
            S(:,nodes_stage(j))=Ptree.Phi{nodes_stage(j)}*Y.y(nodes_stage(j),:)'...
                +sum_u+Ptree.sigma{nodes_stage(j)};
            q(:,nodes_stage(j))=Ptree.c{nodes_stage(j)}+Ptree.d{nodes_stage(j)}'*Y.y(nodes_stage(j),:)'+sum_q;
        else
            if(i==sys.Np)
                %sum_q=sys.Ft{j,1}'*qt{1,j};
                %sum_q=qt{1,j};
                S(:,nodes_stage(j))=Ptree.Phi{nodes_stage(j)}*Y.y(nodes_stage(j),:)'...
                    +Ptree.Theta{Tree.children{nodes_stage(j)}-1}*qt{1,j}+Ptree.sigma{nodes_stage(j)};
                q(:,nodes_stage(j))=Ptree.c{nodes_stage(j)}+Ptree.d{nodes_stage(j)}'*Y.y(nodes_stage(j),:)'+...
                    Ptree.f{Tree.children{nodes_stage(j)}-1}'*qt{1,j};
            else
                sum_q=q(:,Tree.children{nodes_stage(j)});
                %sum_u=q(:,Tree.children{nodes_stage(j)});
                S(:,nodes_stage(j))=Ptree.Phi{nodes_stage(j)}*Y.y(nodes_stage(j),:)'...
                    +Ptree.Theta{Tree.children{nodes_stage(j)}-1}*...
                    sum_q+Ptree.sigma{nodes_stage(j)};
                q(:,nodes_stage(j))=Ptree.c{nodes_stage(j)}+Ptree.d{nodes_stage(j)}'*Y.y(nodes_stage(j),:)'+...
                    Ptree.f{Tree.children{nodes_stage(j)}-1}'*sum_q;
            end
        end
        
    end
end
Z.X(:,1)=xinit;
for i=1:Tree.ancestor(Tree.leaves(end))
    Z.U(:,i)=Ptree.K{i}*Z.X(:,i)+S(:,i);
    for j=1:length(Tree.children{i})
        Z.X(:,Tree.children{i}(j))=sys.A{Tree.children{i}(j)}*Z.X(:,i)+...
            sys.B{Tree.children{i}(j)}*Z.U(:,i)+Tree.value(Tree.children{i}(j),:)';
    end
end
Z.S=S;
Q.q=q;
Q.qt=qt;
end



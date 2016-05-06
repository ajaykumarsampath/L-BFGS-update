function [Z,hessian_ops]=Solve_step_direction(obj,D)
% 
% This function calculate the on-line computaiton of the dual gradient on
% the tree. 
% 
% Syntax : [Z,Q]=Solve_step_line_search(obj,Y,xinit);
% 
% INPUT:         Y     :    dual variable
%              xinit   :    Initial state
% 
% Z is the output and containing

sys=obj.SysMat_.sys;
Ptree=obj.Ptree_;
Tree=obj.SysMat_.tree;
Ns=length(Tree.leaves);
Nd=length(Tree.stage);
non_leaf=Nd-Ns;

Z.X=zeros(sys.nx,Tree.leaves(end));
Z.U=zeros(sys.nu,Tree.ancestor(Tree.leaves(end)));
S=zeros(sys.nu,Tree.ancestor(Tree.leaves(end)));

q=zeros(sys.nx,Tree.ancestor(Tree.leaves(end)));
qt=cell(1,length(Tree.leaves));

for i=1:length(Tree.leaves)
    qt{1,i}=D.yt{i,:};
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
                sum_u=sum_u+Ptree.Theta{Tree.children{nodes_stage(j)}(k)-1}*...
                    q(:,Tree.children{nodes_stage(j)}(k));
                sum_q=sum_q+Ptree.f{Tree.children{nodes_stage(j)}(k)-1}'*...
                    q(:,Tree.children{nodes_stage(j)}(k));
            end
            Z.sum_u{nodes_stage(j)}=sum_u;
            S(:,nodes_stage(j))=Ptree.Phi{nodes_stage(j)}*D.y(:,nodes_stage(j))+sum_u;
            q(:,nodes_stage(j))=Ptree.d{nodes_stage(j)}'*D.y(:,nodes_stage(j))+sum_q;
        else
            if(i==sys.Np)
                %sum_q=sys.Ft{j,1}'*qt{1,j};
                %sum_q=qt{1,j};
                S(:,nodes_stage(j))=Ptree.Phi{nodes_stage(j)}*D.y(:,nodes_stage(j))...
                    +Ptree.Theta{Tree.children{nodes_stage(j)}-1}*qt{1,j};
                q(:,nodes_stage(j))=Ptree.d{nodes_stage(j)}'*D.y(:,nodes_stage(j))...
                    +Ptree.f{Tree.children{nodes_stage(j)}-1}'*qt{1,j};
            else
                sum_q=q(:,Tree.children{nodes_stage(j)});
                %sum_u=q(:,Tree.children{nodes_stage(j)});
                S(:,nodes_stage(j))=Ptree.Phi{nodes_stage(j)}*D.y(:,nodes_stage(j))...
                    +Ptree.Theta{Tree.children{nodes_stage(j)}-1}*sum_q;
                q(:,nodes_stage(j))=Ptree.d{nodes_stage(j)}'*D.y(:,nodes_stage(j))...
                    +Ptree.f{Tree.children{nodes_stage(j)}-1}'*sum_q;
            end
        end
    end
end

Z.X(:,1)=zeros(sys.nx,1);
for i=1:Tree.ancestor(Tree.leaves(end))
    Z.U(:,i)=Ptree.K{i}*Z.X(:,i)+S(:,i);
    for j=1:length(Tree.children{i})
        Z.X(:,Tree.children{i}(j))=sys.A{Tree.children{i}(j)}*Z.X(:,i)+...
            sys.B{Tree.children{i}(j)}*Z.U(:,i);
    end
end

Hx=zeros(2*(sys.nx+sys.nu),non_leaf);
Hx_term=cell(Ns,1);
for i=1:non_leaf
    Hx(:,i)=sys.F{i}*Z.X(:,i)+sys.G{i}*Z.U(:,i);
end

for i=1:Ns
    Hx_term{i,1}=sys.Ft{i,1}*Z.X(:,non_leaf+i);
end 

Z.S=S;
hessian_ops.q=q;
hessian_ops.qt=qt;

hessian_ops.Hx=Hx;
hessian_ops.Hx_term=Hx_term;


end




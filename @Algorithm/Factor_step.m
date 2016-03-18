function obj=Factor_step(obj)
% This function computes the factor step for the accelerated
% proximal gradient method. The matrixes calcualted are stored
% in the structure Ptree
%
% We call the function APG_factor_step function.
%

sys=obj.SysMat_.sys;
tree=obj.SysMat_.tree;
V=obj.SysMat_.V;

Ptree=struct('P',cell(1,1),'c',cell(1,1),'d',cell(1,1),'f',cell(1,1),...
    'Phi',cell(1,1),'Theta',cell(1,1),'sigma',cell(1,1));

for i=1:length(tree.leaves)
    Ptree.P{tree.leaves(i),1}=tree.prob(tree.leaves(i))*V.Vf{i};
end

for i=sys.Np+1:-1:1
    if(i==sys.Np+1)
    else
        nodes_stage=find(tree.stage==i-1);
        total_nodes=length(nodes_stage);
        for j=1:total_nodes
            Pchld=tree.prob(tree.children{nodes_stage(j)})/tree.prob(nodes_stage(j));
            no_child=length(Pchld);
            Pbar=zeros(sys.nu);
            neta=zeros(sys.nu,1);
            neta_x=zeros(sys.nx,1);
            Kbar=zeros(sys.nu,sys.nx);
            for k=1:no_child
                Pbar=Pbar+sys.B{tree.children{nodes_stage(j)}(k)}'*...
                    Ptree.P{tree.children{nodes_stage(j)}(k)}*...
                    sys.B{tree.children{nodes_stage(j)}(k)};% \bar{P}
                neta=neta+sys.B{tree.children{nodes_stage(j)}(k)}'*...
                    Ptree.P{tree.children{nodes_stage(j)}(k)}...
                    *tree.value(tree.children{nodes_stage(j)}(k),:)';%phi_{k-1}^{(i)}
                neta_x=neta_x+sys.A{tree.children{nodes_stage(j)}(k)}'*...
                    Ptree.P{tree.children{nodes_stage(j)}(k)}...
                    *tree.value(tree.children{nodes_stage(j)}(k),:)';%phi_{k-1}^{(i)}
                Kbar=Kbar+sys.B{tree.children{nodes_stage(j)}(k)}'*...
                    Ptree.P{tree.children{nodes_stage(j)}(k)}*...
                    sys.A{tree.children{nodes_stage(j)}(k)};
            end
            
            %terms in the control u_{k-1}^{\star (i)}
            Rbar=2*(tree.prob(nodes_stage(j))*V.R+Pbar);
            Rbar_inv=Rbar\eye(sys.nu);
            Ptree.sigma{nodes_stage(j)}=-2*Rbar_inv*neta;%sigma_{k-1}^{(i)}
            Ptree.K{nodes_stage(j)}=-2*Rbar_inv*Kbar;%K_{k-1}^{(i)}
            Ptree.Phi{nodes_stage(j)}=-Rbar_inv*sys.G{nodes_stage(j)}';%\Phi_{k-1}^{(i)}
            
            if(i==sys.Np)
                for k=1:no_child
                    Ptree.Theta{tree.children{nodes_stage(j)}(k)-1}=-Rbar_inv*...
                        sys.B{tree.children{nodes_stage(j)}(k)}'*sys.Ft{j,1}';%\Theta_{k-1}^{(i)}
                end
            else
                for k=1:no_child
                    Ptree.Theta{tree.children{nodes_stage(j)}(k)-1}=-Rbar_inv*...
                        sys.B{tree.children{nodes_stage(j)}(k)}';%\Theta_{k-1}^{(i)}
                end
            end
            
            
            %terms in the linear cost
            Ptree.c{nodes_stage(j)}=2*(neta_x+Ptree.K{nodes_stage(j)}'*neta);
            
            Ptree.d{nodes_stage(j)}=sys.F{nodes_stage(j)}+sys.G{nodes_stage(j)}*Ptree.K{nodes_stage(j)};%d_{k-1}^{(i)}
            
            if(i==sys.Np)
                for k=1:no_child
                    Ptree.f{tree.children{nodes_stage(j)}(k)-1}=sys.Ft{j,1}*...
                        (sys.A{tree.children{nodes_stage(j)}(k)}+...
                        sys.B{tree.children{nodes_stage(j)}(k)}*Ptree.K{nodes_stage(j)});%f_{k-1}^{(i)}
                end
            else
                for k=1:no_child
                    Ptree.f{tree.children{nodes_stage(j)}(k)-1}=(sys.A{tree.children{nodes_stage(j)}(k)}+...
                        sys.B{tree.children{nodes_stage(j)}(k)}*Ptree.K{nodes_stage(j)});%f_{k-1}^{(i)}
                end
            end
            
            
            %Quadratic cost
            if(i==sys.Np)
                Ptree.P{nodes_stage(j)}=tree.prob(nodes_stage(j))*(V.Q+Ptree.K{nodes_stage(j)}'*V.R*Ptree.K{nodes_stage(j)});
                for k=1:no_child
                    Ptree.P{nodes_stage(j)}=Ptree.P{nodes_stage(j)}+(sys.A{tree.children{nodes_stage(j)}(k)}+...
                        sys.B{tree.children{nodes_stage(j)}(k)}*Ptree.K{nodes_stage(j)})'*...
                        Ptree.P{tree.children{nodes_stage(j)}(k)}*...
                        (sys.A{tree.children{nodes_stage(j)}(k)}+sys.B{tree.children{nodes_stage(j)}(k)}*Ptree.K{nodes_stage(j)});
                end
            else
                Ptree.P{nodes_stage(j)}=tree.prob(nodes_stage(j))*(V.Q+Ptree.K{nodes_stage(j)}'*V.R*Ptree.K{nodes_stage(j)});
                for k=1:no_child
                    Ptree.P{nodes_stage(j)}=Ptree.P{nodes_stage(j)}+Ptree.f{tree.children{nodes_stage(j)}(k)-1}'...
                        *Ptree.P{tree.children{nodes_stage(j)}(k)}*Ptree.f{tree.children{nodes_stage(j)}(k)-1};
                end
                
            end
        end
    end
    
end

obj.Ptree_=Ptree;
end


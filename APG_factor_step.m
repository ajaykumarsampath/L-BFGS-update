function [Ptree] = APG_factor_step( sys,V,Tree)
% This is the Factor step of the algorithm. It calcualtes the offline
% matrices used in the factor step.
% 
% INPUT:    sys       : system dynamics 
%           V         : cost function (normally a quadratic cost)
%           Tree      : The tree structure
%
% OUTPUT:   Ptree     : contains the off-line matrices
%

Ptree=struct('P',cell(1,1),'c',cell(1,1),'d',cell(1,1),'f',cell(1,1),...
    'Phi',cell(1,1),'Theta',cell(1,1),'sigma',cell(1,1));

for i=1:length(Tree.leaves)
    Ptree.P{Tree.leaves(i),1}=Tree.prob(Tree.leaves(i))*V.Vf{i};
end

for i=sys.Np+1:-1:1
    if(i==sys.Np+1)
    else
        nodes_stage=find(Tree.stage==i-1);
        total_nodes=length(nodes_stage);
        for j=1:total_nodes
            Pchld=Tree.prob(Tree.children{nodes_stage(j)})/Tree.prob(nodes_stage(j));
            no_child=length(Pchld);
            Pbar=zeros(sys.nu);
            neta=zeros(sys.nu,1);
            neta_x=zeros(sys.nx,1);
            Kbar=zeros(sys.nu,sys.nx);
            for k=1:no_child
                Pbar=Pbar+sys.B{Tree.children{nodes_stage(j)}(k)}'*...
                    Ptree.P{Tree.children{nodes_stage(j)}(k)}*...
                    sys.B{Tree.children{nodes_stage(j)}(k)};% \bar{P}
                neta=neta+sys.B{Tree.children{nodes_stage(j)}(k)}'*...
                    Ptree.P{Tree.children{nodes_stage(j)}(k)}...
                    *Tree.value(Tree.children{nodes_stage(j)}(k),:)';%phi_{k-1}^{(i)}
                neta_x=neta_x+sys.A{Tree.children{nodes_stage(j)}(k)}'*...
                    Ptree.P{Tree.children{nodes_stage(j)}(k)}...
                    *Tree.value(Tree.children{nodes_stage(j)}(k),:)';%phi_{k-1}^{(i)}
                Kbar=Kbar+sys.B{Tree.children{nodes_stage(j)}(k)}'*...
                    Ptree.P{Tree.children{nodes_stage(j)}(k)}*...
                    sys.A{Tree.children{nodes_stage(j)}(k)};
            end
            
            %terms in the control u_{k-1}^{\star (i)}
            Rbar=2*(Tree.prob(nodes_stage(j))*V.R+Pbar);
            Rbar_inv=Rbar\eye(sys.nu);
            Ptree.sigma{nodes_stage(j)}=-2*Rbar_inv*neta;%sigma_{k-1}^{(i)}
            Ptree.K{nodes_stage(j)}=-2*Rbar_inv*Kbar;%K_{k-1}^{(i)}
            Ptree.Phi{nodes_stage(j)}=-Rbar_inv*sys.G{nodes_stage(j)}';%\Phi_{k-1}^{(i)}
            
            if(i==sys.Np)
                for k=1:no_child
                    Ptree.Theta{Tree.children{nodes_stage(j)}(k)-1}=-Rbar_inv*...
                        sys.B{Tree.children{nodes_stage(j)}(k)}'*sys.Ft{j,1}';%\Theta_{k-1}^{(i)}
                end
            else
                for k=1:no_child
                    Ptree.Theta{Tree.children{nodes_stage(j)}(k)-1}=-Rbar_inv*...
                        sys.B{Tree.children{nodes_stage(j)}(k)}';%\Theta_{k-1}^{(i)}
                end
            end
            
            
            %terms in the linear cost
            Ptree.c{nodes_stage(j)}=2*(neta_x+Ptree.K{nodes_stage(j)}'*neta);
            
            Ptree.d{nodes_stage(j)}=sys.F{nodes_stage(j)}+sys.G{nodes_stage(j)}*Ptree.K{nodes_stage(j)};%d_{k-1}^{(i)}
            
            if(i==sys.Np)
                for k=1:no_child
                    Ptree.f{Tree.children{nodes_stage(j)}(k)-1}=sys.Ft{j,1}*...
                        (sys.A{Tree.children{nodes_stage(j)}(k)}+...
                        sys.B{Tree.children{nodes_stage(j)}(k)}*Ptree.K{nodes_stage(j)});%f_{k-1}^{(i)}
                end
            else
                for k=1:no_child
                    Ptree.f{Tree.children{nodes_stage(j)}(k)-1}=(sys.A{Tree.children{nodes_stage(j)}(k)}+...
                        sys.B{Tree.children{nodes_stage(j)}(k)}*Ptree.K{nodes_stage(j)});%f_{k-1}^{(i)}
                end
            end
            
            
            %Quadratic cost
            if(i==sys.Np)
                Ptree.P{nodes_stage(j)}=Tree.prob(nodes_stage(j))*(V.Q+Ptree.K{nodes_stage(j)}'*V.R*Ptree.K{nodes_stage(j)});
                for k=1:no_child
                    Ptree.P{nodes_stage(j)}=Ptree.P{nodes_stage(j)}+(sys.A{Tree.children{nodes_stage(j)}(k)}+...
                        sys.B{Tree.children{nodes_stage(j)}(k)}*Ptree.K{nodes_stage(j)})'*...
                        Ptree.P{Tree.children{nodes_stage(j)}(k)}*...
                        (sys.A{Tree.children{nodes_stage(j)}(k)}+sys.B{Tree.children{nodes_stage(j)}(k)}*Ptree.K{nodes_stage(j)});
                end
            else
                Ptree.P{nodes_stage(j)}=Tree.prob(nodes_stage(j))*(V.Q+Ptree.K{nodes_stage(j)}'*V.R*Ptree.K{nodes_stage(j)});
                for k=1:no_child
                    Ptree.P{nodes_stage(j)}=Ptree.P{nodes_stage(j)}+Ptree.f{Tree.children{nodes_stage(j)}(k)-1}'...
                        *Ptree.P{Tree.children{nodes_stage(j)}(k)}*Ptree.f{Tree.children{nodes_stage(j)}(k)-1};
                end
                
            end
        end
    end
    
end

end




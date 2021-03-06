function [Z,Y1,details]=Dual_APG(obj,x0)
% This function calculate the optimal solution using the
% APG algorithm on the the dual problem for the system at the
% given initial point
%
% INPUT:     x0  :  initial point
%
% OUTPUT:    Z   :  primal solution
%            Y   :  dual solution
%       details  :  structure containing details like time of
%                   computation, number of itrataions.
%

sys=obj.SysMat_.sys;
Tree=obj.SysMat_.tree;
V=obj.SysMat_.V;
ops=obj.algo_details.ops_APG;


Ns=length(Tree.leaves); % total scenarios in the tree
Nd=length(Tree.stage); %  toal nodes in the tree

% Initalizing the dual varibables
Y0.y=zeros(size(sys.F{1},1),Nd-Ns);
Y1.y=zeros(size(sys.F{1},1),Nd-Ns);

dual_grad_term=cell(Ns,1);
%epsilon_prm=1;

for i=1:Ns
    Y0.yt{i,:}=zeros(size(sys.Ft{i,1},1),1);
    Y1.yt{i,:}=zeros(size(sys.Ft{i,1},1),1);
    dual_grad_term{i,1}=zeros(size(sys.Ft{i,1},1),1);
end

%prm_fes=zeros(size(sys.F{1},1),Nd-Ns);
g_nodes=zeros(size(sys.F{1},1),Nd-Ns);
for i=1:Nd-Ns
    g_nodes(:,i)=sys.g{i};
end
g_nodes_term=sys.gt;
theta=[1 1]';
tic
j=1;

W_minyt=zeros(Ns,1);

details.term_crit=zeros(1,4);
%dual_grad=prm_fes;
%dual_grad_term=prm_fes_term;

while(j<ops.steps)
    % Step 1: accelerated step
    W.y=Y1.y+theta(2)*(1/theta(1)-1)*(Y1.y-Y0.y);
    
    for i=1:Ns
        W.yt{i,1}=Y1.yt{i,1}+theta(2)*(1/theta(1)-1)*(Y1.yt{i,1}-Y0.yt{i,1});
        W_minyt(i,1)=min(W.yt{i,1});
    end
    
    
    %step 2: dual gradient calculation 
    Z=obj.Solve_step(W,x0);
    
    %step 3: Projection of y on the positive quadrant.
    Y0.y=Y1.y;
    Y0.yt=Y1.yt;
    
    [Y1,details_prox_gconj]=obj.proximal_gconj(Z,W);
    
    details.lambda(j)=details_prox_gconj.lambda;
    obj.algo_details.ops_APG.lambda=details_prox_gconj.lambda;
    
    %dual_grad=details_prox_gconj.dual_grad;
    prm_fes=details_prox_gconj.prm_fes;
    
    %dual_grad_term=details_prox_gconj.dual_grad_term;
    prm_fes_term=details_prox_gconj.prm_fes_term;
    
    details.prm_fes{j}=prm_fes;
    
    iter=j;
    details.prm_cst(iter)=0;%primal cost;
    details.dual_cst(iter)=0;% dual cost;
    
    %termination criteria 
    %{
    epsilon=zeros(Nd,1);
    for i=1:Nd-Ns   
        epsilon(i,1)=max(max(sys.F{i}*Z.X(:,i)+sys.G{i}*Z.U(:,i)-sys.g{i},0));
    end
    
    for i=1:Ns
        j=Tree.leaves(i);     
        epsilon(j,1)=max(max(sys.Ft{i}*Z.X(:,j)-sys.gt{i},0));
    end
    
    primal_epsilon=max(epsilon)
    %}
    epsilon_prm=max(max(max(prm_fes,0)));
    epsilon_prm=max(max(cell2mat(prm_fes_term),epsilon_prm));
    
    if(epsilon_prm>ops.primal_inf)
        % step 4: theta update
        if(strcmp(obj.algo_details.ops_APG.type,'yes'))
            theta(1)=theta(2);
            theta(2)=(sqrt(theta(1)^4+4*theta(1)^2)-theta(1)^2)/2;
        end 
        j=j+1;
    else
        details.iterate=j;
        break
    end
    
    %{
    termination criteria
    if(j==1)
        prm_avg_next=dual_grad;
        prm_avg_term_next=dual_grad_term;
        epsilon_prm_avg=max( max(max(dual_grad-g_nodes)), ...
            max(max(cell2mat(dual_grad_term)-cell2mat(g_nodes_term))) );
    else
        prm_avg_next=(1-theta(2))*prm_avg_next+theta(2)*dual_grad;
        for m=1:Ns
            prm_avg_term_next{m,1}=(1-theta(2))*prm_avg_term_next{m,1}+theta(2)*dual_grad_term{m,1};
        end
        epsilon_prm_avg=max(max(max(prm_avg_next-g_nodes)),...
            max(max(cell2mat(prm_avg_term_next)-cell2mat(g_nodes_term))));
    end
    if epsilon_prm_avg<=ops.primal_inf %average_primal feasibility less
        details.term_crit(1,2)=1;
        details.iterate=j;
        j=10*ops.steps;
    else
        epsilon_prm=max( max(max(dual_grad-g_nodes)), ...
            max(max(cell2mat(dual_grad_term)-cell2mat(g_nodes_term))) );
        if(epsilon_prm<=ops.primal_inf) % primal feasibility of the iterate
            if (min(min(min(W.y)),min( W_minyt))>0)
                sum=0;
                for i=1:Nd-Ns
                    sum=sum-W.y(i,:)*(dual_grad(:,i)-g_nodes(:,i));
                end
                for i=1:Ns
                    sum=sum-W.yt{i,:}*(dual_grad_term{i,1}-g_nodes_term{i,1});
                end
                if sum<=ops.dual_gap %condition 29. dual gap
                    details.term_crit(1,2)=1;
                    details.iterate=j;
                    j=10*ops.steps;
                else
                    prm_cst=0;%primal cost;
                    for i=1:Nd-Ns
                        prm_cst=prm_cst+Tree.prob(i,1)*(Z.X(:,i)'*V.Q*Z.X(:,i)+Z.U(:,i)'*V.R*Z.U(:,i));
                    end
                    for i=1:Ns
                        prm_cst=prm_cst+Tree.prob(Tree.leaves(i))*(Z.X(:,Tree.leaves(i))'*V.Vf{i,1}*...
                            Z.X(:,Tree.leaves(i)));
                    end
                    if sum<=ops.dual_gap*prm_cst/(1+ops.dual_gap) %condition 30 dual gap
                        details.term_crit(1,3)=1;
                        details.iterate=j;
                        j=10*ops.steps;
                    else
                        %step 4: theta update
                        theta(1)=theta(2);
                        theta(2)=(sqrt(theta(1)^4+4*theta(1)^2)-theta(1)^2)/2;
                        j=j+1;
                    end
                end
            else
                prm_cst=0;%primal cost;
                dual_cst=0;% dual cost;concussion
                for i=1:Nd-Ns
                    prm_cst=prm_cst+Tree.prob(i,1)*(Z.X(:,i)'*V.Q*Z.X(:,i)+Z.U(:,i)'*V.R*Z.U(:,i));
                    dual_cst=dual_cst+Y1.y(:,i)'*(dual_grad(:,i)-g_nodes(:,i));
                end
                for i=1:Ns
                    prm_cst=prm_cst+Tree.prob(Tree.leaves(i))*(Z.X(:,Tree.leaves(i))'*V.Vf{i,1}*...
                        Z.X(:,Tree.leaves(i)));
                    dual_cst=dual_cst+Y1.yt{i}'*(dual_grad_term{i,1}-g_nodes_term{i,1});
                end
                if (-dual_cst<=ops.dual_gap*max(dual_cst,1)) %condtion 27 (dual gap)
                    details.term_crit(1,4)=1;
                    details.iterate=j;
                    j=10*ops.steps;
                else
                    %step 4: theta update
                    theta(1)=theta(2);
                    theta(2)=(sqrt(theta(1)^4+4*theta(1)^2)-theta(1)^2)/2;
                    j=j+1;
                end
            end
        else
            %step 4: theta update
            theta(1)=theta(2);
            theta(2)=(sqrt(theta(1)^4+4*theta(1)^2)-theta(1)^2)/2;
            j=j+1;
        end
    end
    %}
    details.epsilon_prm(iter)=epsilon_prm;
    details.dual_grad(iter)=0;
    for i=1:Nd-Ns
        details.prm_cst(iter)=details.prm_cst(iter)+Tree.prob(i,1)*(Z.X(:,i)'*V.Q*Z.X(:,i)+Z.U(:,i)'*V.R*Z.U(:,i));
        details.dual_grad(iter)=details.dual_grad(iter)+(prm_fes(:,i))'*(Y1.y(:,i)-Y0.y(:,i));
        details.dual_cst(iter)=details.dual_cst(iter)+Y1.y(:,i)'*(prm_fes(:,i));
    end
    for i=1:Ns
        details.prm_cst(iter)=details.prm_cst(iter)+Tree.prob(Tree.leaves(i))*(Z.X(:,Tree.leaves(i))'*V.Vf{i,1}*...
            Z.X(:,Tree.leaves(i)));
        details.dual_grad(iter)=details.dual_grad(iter)+(Y1.yt{i,1}-Y0.yt{i,1})'*(dual_grad_term{i,1}-g_nodes_term{i,1});
        details.dual_cst(iter)=details.dual_cst(iter)+Y1.yt{i,1}'*(prm_fes_term{i,1});
    end
    details.dual_cst(iter)=details.prm_cst(iter)+details.dual_cst(iter);
    
end
details.dual_gap=details.prm_cst(iter)-details.dual_cst(iter);
details.gpad_solve=toc;
details.W=W;
%details.epsilon_prm_avg= epsilon_prm_avg;
%details.epsilon_prm=epsilon_prm;


end
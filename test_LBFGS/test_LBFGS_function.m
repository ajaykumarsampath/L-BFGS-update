% This function tests the L-BFGS function implementation
% We generate a random QP: 0.5*x'Qx+b'x
%%
clc;
clear all;
close all;
size_mat=1000;
lb=-5*ones(size_mat,1);
ub=5*ones(size_mat,1);
Q=20*(rand(size_mat));
Q=Q'*Q;
if(min(eig(Q))<0)
    disp('error')
else
    b=20*(rand(size_mat,1)-0.4);
    
    % actual solution
    Xmin= -Q\b;
    value_temp=0.5*Xmin'*Q*Xmin+b'*Xmin;
    
    
    %% gradient soution
    max_iter=8000;
    tot_iter=max_iter-1;
    Xiter=zeros(size_mat,max_iter);
    value_grad=zeros(max_iter-1,1);
    iter=1;
    step_size=1/norm(Q,2);
    Xiter(:,1)=Xmin+0.9*Xmin;
    while(iter<max_iter)
        Xiter(:,iter+1)=Xiter(:,iter)-step_size*(Q*Xiter(:,iter)+b);
        %{
    for j=1:size_mat
        if(Xiter(j,iter+1)>ub(j))
            Xiter(j,iter+1)=ub(j);
        elseif(Xiter(j,iter+1)<lb(j))
            Xiter(j,iter+1)=lb(j);
        else
            Xiter(j,iter+1)=Xiter(j,iter+1);
        end
    end
        %}
        value_grad(iter,1)=0.5*Xiter(:,iter+1)'*Q*Xiter(:,iter+1)+b'*Xiter(:,iter+1);
        
        if(iter>2)
            if(abs(value_grad(iter)-value_grad(iter-1))<1e-4)
                tot_iter=iter;
                iter=max_iter+1;
            else
                iter=iter+1;New Folder
            end
        else
            iter=iter+1;
        end
        %{
    if(norm(Xiter(:,iter+1)-Xiter(:,iter),inf)<1e-6)
        tot_iter=iter;
        iter=max_iter+1;
    else
        iter=iter+1;
    end
        %}
        %
    end
    figure(1)
    hold all;
    plot(value_grad(1:tot_iter))
    %% Newton method
    inv_Q=Q\eye(size_mat);
    
    max_iter_new=8000;
    tot_iter_new=max_iter_new-1;
    Xiter_new=zeros(size_mat,max_iter_new);
    value_grad_new=zeros(max_iter_new-1,1);
    iter=1;
    step_size=1/norm(Q,2);
    Xiter_new(:,1)=Xmin+0.9*Xmin;
    while(iter<max_iter_new)
        Xiter_new(:,iter+1)=Xiter_new(:,iter)-step_size*inv_Q*(Q*Xiter_new(:,iter)+b);
        value_grad_new(iter,1)=0.5*Xiter_new(:,iter+1)'*Q*Xiter_new(:,iter+1)+b'*Xiter_new(:,iter+1);
        
        if(iter>2)
            if(abs(value_grad_new(iter)-value_grad_new(iter-1))<1e-4)
                tot_iter_new=iter;
                iter=max_iter_new+1;
            else
                iter=iter+1;
            end
        else
            iter=iter+1;
        end
    end
    figure(1)
    hold all;
    plot(value_grad_new(1:tot_iter_new))
    %%
    Xyalmip = sdpvar(size_mat,1);
    ops_grb = sdpsettings('verbose',0,'solver','gurobi');
    ops = sdpsettings('verbose',0);
    objective=0.5*Xyalmip'*Q*Xyalmip+b'*Xyalmip;
    constraints=[lb<=Xyalmip<=ub];
    
    normal_optimise=optimize(constraints,objective,ops_grb);%,ops,{},solutions_out);%,ops,{},solutions_out);
    gurobi_opt=optimize([],objective,ops_grb);
    
    l=value(Xyalmip);
    %obj=0.5*l'*Q*l+b'*l-value(objective)
    %
    value(objective)-value_grad(tot_iter)
    value(objective)-value_grad_new(tot_iter_new)
    norm(l-Xiter(:,tot_iter),inf)
end


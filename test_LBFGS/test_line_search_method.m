% This function tests the L-BFGS function implementation
% We generate a random QP: 0.5*x'Qx+b'x
%%
clc;
clear all;
close all;
size_mat=300;
Q=20*(rand(size_mat));
Q=Q'*Q;
if(min(eig(Q))<0)
    disp('error')
else
    b=20*(rand(size_mat,1)-0.4);
    
    % actual solution
    Xmin= -Q\b;
    value_temp=0.5*Xmin'*Q*Xmin+b'*Xmin;
    
    fun_quad.Q=Q;
    fun_quad.b=b;
    
    ops_line_search.iter_max=500;
    
    %% gradient soution with line search
    max_iter=6000;
    tot_iter=max_iter-1;
    alphas=zeros(3,max_iter);
    Xiter=zeros(size_mat,max_iter);
    value_grad=zeros(max_iter-1,1);
    iter=1;
    %Xiter(:,1)=Xmin+0.9*Xmin;
    while(iter<max_iter)
        Grad=(Q*Xiter(:,iter)+b);
        d=-Grad; % direction
        alpha_size=wolf_linesearch( fun_quad,Grad,Xiter(:,iter),d,ops_line_search);
        Xiter(:,iter+1)=Xiter(:,iter)+alpha_size*d;
        alphas(1,iter)=alpha_size;
        %Xiter(:,iter+1)=Xiter(:,iter)-alpha_size*(Q*Xiter(:,iter)+b);
        value_grad(iter,1)=0.5*Xiter(:,iter+1)'*Q*Xiter(:,iter+1)+b'*Xiter(:,iter+1);
        
        if(iter>2)
            %if(abs(value_grad(iter)-value_grad(iter-1))<1e-4)
            if(norm(Grad)<1e-8)
                tot_iter=iter;
                iter=max_iter+1;
            else
                iter=iter+1;
            end
        else
            iter=iter+1;
        end
    end
    %% Gradient method without line search
    
    tot_iter_ls=max_iter-1;
    Xiter_wls=zeros(size_mat,max_iter);
    value_grad_wls=zeros(max_iter-1,1);
    iter=1;
    step_size=1/norm(Q,2);
    %Xiter(:,1)=Xmin+0.9*Xmin;
    while(iter<max_iter)
        Grad=(Q*Xiter_wls(:,iter)+b);
        d=-Grad; % direction
        Xiter_wls(:,iter+1)=Xiter_wls(:,iter)-step_size*(Q*Xiter_wls(:,iter)+b);
        value_grad_wls(iter,1)=0.5*Xiter_wls(:,iter+1)'*Q*Xiter_wls(:,iter+1)+b'*Xiter_wls(:,iter+1);
        
        if(iter>2)
            %if(abs(value_grad(iter)-value_grad(iter-1))<1e-4)
            if(norm(Grad)<1e-4)
                tot_iter_ls=iter;
                iter=max_iter+1;
            else
                iter=iter+1;
            end
        else
            iter=iter+1;
        end
    end
    
    %% Newton method
    inv_Qb=Q\b;
    %inv_Q=inv(Q);
    
    tot_iter_new=max_iter-1;
    Xiter_new=zeros(size_mat,max_iter);
    value_grad_new=zeros(max_iter-1,1);
    iter=1;
    %Xiter_new(:,1)=Xmin+0.9*Xmin;
    while(iter<max_iter)
        Grad=(Q*Xiter_new(:,iter)+b);
        d=-(Xiter_new(:,iter)+inv_Qb);
        alpha_size=wolf_linesearch( fun_quad,Grad,Xiter_new(:,iter),d,ops_line_search);
        Xiter_new(:,iter+1)=Xiter_new(:,iter)-alpha_size*(Xiter_new(:,iter)+inv_Qb);
        %Xiter_new(:,iter+1)=Xiter_new(:,iter)-splot(value_grad(1:tot_iter))tep_size*(Xiter_new(:,iter)+inv_Qb);
        value_grad_new(iter,1)=0.5*Xiter_new(:,iter+1)'*Q*Xiter_new(:,iter+1)+b'*Xiter_new(:,iter+1);
        
        if(iter>2)
            %if(abs(value_grad_new(iter)-value_grad_new(iter-1))<1e-4)
            if(norm(d)<1e-3)
                tot_iter_new=iter;
                iter=max_iter+1;
            else
                iter=iter+1;
            end
        else
            iter=iter+1;
        end
    end
    %figure(1)
    %hold all;
    %plot(value_grad_new(1:tot_iter_new))
    %% diagonal hessian
    inv_Qdiag=diag(diag(Q\eye(size_mat)));
    %inv_Q=inv(Q);
    tot_iter_diag=max_iter-1;
    Xiter_diag=zeros(size_mat,max_iter);
    value_grad_diag=zeros(max_iter-1,1);
    iter=1;
    %Xiter_new(:,1)=Xmin+0.9*Xmin;
    while(iter<max_iter)
        Grad=(Q*Xiter_diag(:,iter)+b);
        d=-inv_Qdiag*Grad;
        alpha_size=wolf_linesearch( fun_quad,Grad,Xiter_diag(:,iter),d,ops_line_search);
        alphas(2,iter)=alpha_size;
        Xiter_diag(:,iter+1)=Xiter_diag(:,iter)+alpha_size*d;
        %Xiter_new(:,iter+1)=Xiter_new(:,iter)-step_size*(Xiter_new(:,iter)+inv_Qb);
        value_grad_diag(iter,1)=0.5*Xiter_diag(:,iter+1)'*Q*Xiter_diag(:,iter+1)+b'*Xiter_diag(:,iter+1);
        
        if(iter>2)
            %if(abs(value_grad_new(iter)-value_grad_new(iter-1))<1e-4)
            if(norm(d)<1e-8)
                tot_iter_diag=iter;
                iter=max_iter+1;
            else
                iter=iter+1;
            end
        else
            iter=iter+1;
        end
    end
    
    %% L-BFGS algorithm
    memory=10;
    Lbfgs.S  = zeros(size_mat, memory);
    Lbfgs.Y  = zeros(size_mat, memory);
    Lbfgs.YS = zeros(memory, 1);
    Lbfgs.LBFGS_col = 1;Lbfgs.LBFGS_mem = 0;Lbfgs.skipCount = 0;
    alphaC=1;
    skipCount=0;
    tot_iter_Lbfgs=max_iter-1;
    Xiter_LBFGS=zeros(size_mat,max_iter);
    value_grad_LBFGS=zeros(max_iter-1,1);
    iter=1;
    
    while(iter<max_iter) 
        Grad=(Q*Xiter_LBFGS(:,iter)+b);
        if(iter==1)
            d=-Grad;
            LBFGS_col = 1;
            LBFGS_mem = 0;
        else
            if(norm(Grad-GradOld)>0)
                Sk  = (Xiter_LBFGS(:,iter) - Xiter_LBFGS(:,iter-1));
                Yk  = Grad - GradOld;
                YSk = Yk'*Sk;
                if norm(Grad) < 1,alphaC = 3;end
                if YSk/(Sk'*Sk) > 1e-6*norm(Grad) ^alphaC
                    LBFGS_col = 1 + mod(LBFGS_col, memory);
                    LBFGS_mem = min(LBFGS_mem+1, memory);
                    Lbfgs.S(:,LBFGS_col) = Sk;
                    Lbfgs.Y(:,LBFGS_col) = Yk;
                    Lbfgs.YS(LBFGS_col)  = YSk;
                else
                    skipCount = skipCount+1;
                end
                Lbfgs.H = YSk/(Yk'*Yk);
                d = LBFGS(Lbfgs.S,Lbfgs.Y,Lbfgs.YS,Lbfgs.H,...
                    -Grad, int32(LBFGS_col), int32(LBFGS_mem));
            else
                d=zeros(size_mat,1);
            end

        end
        grad_value(iter)=Grad'*d;
        alpha_size=wolf_linesearch( fun_quad,Grad,Xiter_LBFGS(:,iter),d,ops_line_search);
        alphas(3,iter)=alpha_size;
        Xiter_LBFGS(:,iter+1)=Xiter_LBFGS(:,iter)+alpha_size*d;
        
        value_grad_LBFGS(iter,1)=0.5*Xiter_LBFGS(:,iter+1)'*Q*Xiter_LBFGS(:,iter+1)...
            +b'*Xiter_LBFGS(:,iter+1);
        GradOld=Grad;
        if(iter>2)
            if(norm(d)<1e-8 || alpha_size==0)
                tot_iter_Lbfgs=iter;
                iter=max_iter+1;
            else
                iter=iter+1;
            end
        else
            iter=iter+1;
        end  
    end
    
    figure(3)
    hold all;
    plot(value_grad_LBFGS(1:tot_iter_Lbfgs))
    max(grad_value)
end

%%
figure(1)
hold all;
plot(value_grad(1:tot_iter))
plot(value_grad_wls(1:tot_iter_ls))
plot(value_grad_diag(1:tot_iter_diag))
plot(value_grad_LBFGS(1:tot_iter_Lbfgs))
legend('grad-ls','grad','diag-Hessian','LBFGS')
%%
%
Xyalmip = sdpvar(size_mat,1);
ops_grb = sdpsettings('verbose',0,'solver','gurobi');
ops = sdpsettings('verbose',0);
objective=0.5*Xyalmip'*Q*Xyalmip+b'*Xyalmip;
%constraints=[lb<=Xyalmip<=ub];

%normal_optimise=optimize(constraints,objective,ops_grb);%,ops,{},solutions_out);%,ops,{},solutions_out);
gurobi_opt=optimize([],objective,ops_grb);

l=value(Xyalmip);
%obj=0.5*l'*Q*l+b'*l-value(objective)
%
[value(objective)-value_grad(tot_iter),
    value(objective)-value_grad_new(tot_iter_new),
    value(objective)-value_grad_diag(tot_iter_diag),
    value(objective)-value_grad_LBFGS(tot_iter_Lbfgs)]'

figure(2)
plot(alphas')

%{
    norm(l-Xiter(:,tot_iter),inf),
    norm(l-Xiter_diag(:,tot_iter_diag),inf),
    norm(l-Xiter_LBFGS(:,tot_iter_Lbfgs),inf)
%}
    
%%
% This script solves optimisation of the scenario-tree based model
% predictive control. The dynamic system considered is the standard
% spring-masss system.
%%
% Generation of the system

clear all;
close all;
clear model;
clc;
Nm=2; % Number of masses
T_sampling=0.5;
ops_masses=struct('Nm',Nm,'Ts',T_sampling,'xmin', ...
    -4*ones(2*Nm,1), 'xmax', 4*ones(2*Nm,1), 'umin', -2*ones(Nm-1,1),'umax',...
    2*ones(Nm-1,1),'b', 0.1*ones(Nm+1,1));

ops_system.brch_fact=[5 5 3]; % branching factor of the tree
ops_system.uncertainty='additive'; % kind of uncertainty; 0 for additive; 1 for additive
...and parametric
ops_system.N=5; % prediction horizon

SysMat=SysMat(ops_masses,ops_system);
sys_actual=SysMat.sys;

SysMat.sys=scale_constraints_system(SysMat);

% Jacobi 
SysMat_FBE_prcnd=SysMat;
SysMat_FBE_prcnd.sys_ops.precondition='Jacobi';
SysMat_FBE_prcnd=SysMat_FBE_prcnd.Precondition_system();
% SysMat.sys=scale_constraints_system(SysMat);

% simple 
SysMat_FBE_simple=SysMat;
SysMat_FBE_simple.sys_ops.precondition='simple';
SysMat_FBE_simple=SysMat_FBE_simple.Precondition_system();

%cond_num(2)=calculate_condition_number(SysMat_FBE_prcnd);
%cond_num(1)=calculate_condition_number(SysMat);
%cond_num(3)=calculate_condition_number(SysMat_FBE_simple);

% simple-Proximal gradient method
SysMat_FBE=SysMat;
SysMat_FBE.sys_ops.precondition='no';
% test points
test_points=50;
%% Algorithm classes- FBE
AlgoFBE_options.ops_FBE.steps=100;
AlgoFBE_options.ops_FBE.epsilon=1e-6;
AlgoFBE_options.algorithm='FBE';
AlgoFBE_options.ops_FBE.memory=5;
AlgoFBE_options.ops_FBE.lambda=0.1;
%AlgoFBE_opgrad_dual_enveloptions.ops_FBE.LS='GOLDSTEIN';

% FBE algorithm-precondition Jacobi
%AlgoFBE_options.ops_FBE.lambda=0.1;
FBE_algo_prcnd=Algorithm(SysMat_FBE_prcnd,AlgoFBE_options);
%FBE_algo_prcnd.algo_details.ops_FBE.lambda=0.1;
% Factor step
FBE_algo_prcnd=FBE_algo_prcnd.Factor_step();

% FBE algorithm-precondition simple
AlgoFBE_options.ops_FBE.lambda=0.1;
FBE_algo_simple=Algorithm(SysMat_FBE_simple,AlgoFBE_options);
%FBE_algo_prcnd.algo_details.ops_FBE.lambda=0.1;
% Factor step
FBE_algo_simple=FBE_algo_simple.Factor_step();

% Algorithm-FBE
FBE_algo=Algorithm(SysMat_FBE,AlgoFBE_options);
FBE_algo=FBE_algo.Factor_step();

%% Gurobi algorithm-model

params.outputflag=0;

tic
model=gurobi_solve(sys_actual,SysMat.V,SysMat.tree);
time_gurobi=toc;


%% Algorithm testing
Xtest=4*rand(SysMat.sys.nx,test_points)-2;
iterates=zeros(test_points,4);
mean_inner_iter=zeros(test_points,3);
inner_final_iter=zeros(test_points,3);
%seper_value=zeros(2,test_points);
cost_function=zeros(test_points,5);
primal_epsilon=zeros(test_points,5);
%
for i=1:test_points
    ops_FBE.x0=Xtest(:,i);
    %%
    % FBE preconditioned 
    %FBE_algo_prcnd.algo_details.ops_FBE.lambda=0.1;
    FBE_algo_prcnd.algo_details.ops_FBE.prox_LS='yes';
    [Zclass_FBE_prcnd,Yclass_FBE_prcnd,Details_FBE_prcnd]=FBE_algo_prcnd.Dual_FBE(ops_FBE.x0);
    %[Zclass_FBE_prcnd,Yclass_FBE_prcnd,Details_FBE_prcnd]=FBE_algo_prcnd.Dual_FBE(ops_FBE.x0);
   
    iterates(i,1)=Details_FBE_prcnd.iter;
    if(Details_FBE_prcnd.iter>1)
        inner_final_iter(i,1)=Details_FBE_prcnd.inner_loops(end);
        if(inner_final_iter(i,1)==10)
            disp('max iterations reached')
            Details_FBE_prcnd.tau
        end
        mean_inner_iter(i,1)=mean(Details_FBE_prcnd.inner_loops);
    else
        mean_inner_iter(i,1)=0;
    end
    % simple preconditioned algorithm 
    %FBE_algo_prcnd.algo_details.ops_FBE.lambda=0.0009;
    FBE_algo_simple.algo_details.ops_FBE.prox_LS='yes';
    [Zclass_FBE_simple,Yclass_FBE_simple,Details_FBE_simple]=...
        FBE_algo_simple.Dual_FBE(ops_FBE.x0);
   
    iterates(i,2)=Details_FBE_simple.iter;
    if(Details_FBE_simple.iter>1)
        inner_final_iter(i,2)=Details_FBE_simple.inner_loops(end);
        if(inner_final_iter(i,2)==10)
            disp('max iterations reached')
            Details_FBE_simple.tau
        end
        mean_inner_iter(i,2)=mean(Details_FBE_prcnd.inner_loops);
    else
        mean_inner_iter(i,2)=0;
    end
    
    % FBE algorithm without preconditioning  
    FBE_algo.algo_details.ops_FBE.prox_LS='yes';
    [Zclass_FBE,Yclass_FBE,Details_FBE]=FBE_algo.Dual_FBE(ops_FBE.x0);
    %
    iterates(i,3)=Details_FBE.iter;
    if(Details_FBE.iter>1)
        inner_final_iter(i,3)=Details_FBE.inner_loops(end);
        if(inner_final_iter(i,3)==10)
            disp('max iterations reached')
            Details_FBE.tau
        end
        mean_inner_iter(i,3)=mean(Details_FBE.inner_loops);
    else
        mean_inner_iter(i,3)=0;
    end
    %}
    %  Gurobi
    model.rhs(end-sys_actual.nx+1:end)=ops_FBE.x0;
    %tic
    results=gurobi(model,params);
    %time_solver(1,kk)=toc;
    Zgurobi.X=zeros(sys_actual.nx,length(SysMat.tree.stage));
    Zgurobi.U=zeros(sys_actual.nu,length(sys_actual.F));
    if(strcmp(results.status,'OPTIMAL'))
        disp('OK');
        nz=sys_actual.nx+sys_actual.nu;
        nx=sys_actual.nx;
        nu=sys_actual.nu;
        non_leaf=length(sys_actual.F);
        Ns=length(sys_actual.Ft);
        for j=1:non_leaf
            Zgurobi.X(:,j)=results.x((j-1)*nz+1:(j-1)*nz+nx);
            Zgurobi.U(:,j)=results.x((j-1)*nz+nx+1:j*nz);
        end
        for j=1:Ns
            Zgurobi.X(:,non_leaf+j)=results.x(non_leaf*nz+(j-1)*nx+1:non_leaf*nz+j*nx);
        end
    else
        disp('Error');
    end
    
    %
    

    
    [cost_function(i,1),primal_epsilon(i,1)]=SysMat.cost_function(Zclass_FBE_prcnd);
    
    [cost_function(i,2),primal_epsilon(i,2)]=SysMat.cost_function(Zgurobi);
    [cost_function(i,3),primal_epsilon(i,3)]=SysMat.cost_function(Zclass_FBE_simple);
    
    [cost_function(i,4),primal_epsilon(i,4)]=SysMat.cost_function(Zclass_FBE);
end


%%


%{
% iterations
%plot(iterate);

% plot states
plot(Zclass_FBE.U(3,:))
hold all;
plot(Zclass_FBE2.U(3,:))
plot(Zgurobi.U(3,:))
plot(Zclass_APG.U(3,:))
legend('FBE','FBE2','gurobi','FBE')
%
%norm(Zclass_init.U-Zgurobi.U,inf)
max(max(abs(Zclass_FBE.U-Zgurobi.U)))
max(max(abs(Zclass_FBE2.U-Zgurobi.U)))
max(max(abs(Zclass_APG.U-Zgurobi.U)))
%}
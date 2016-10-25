%%
% This script solves optimisation of the scenario-tree based model
% predictive control. The dynamic system considered is the standard
% spring-masss system--testing with the normal system without any
% preconditioning 
%%
% Generation of the system

clear all;
close all;
clear model;
clc;
Nm=3; % Number of masses
T_sampling=0.5;
ops_masses=struct('Nm',Nm,'Ts',T_sampling,'xmin', ...
    -4*ones(2*Nm,1), 'xmax', 4*ones(2*Nm,1), 'umin', -2*ones(Nm-1,1),'umax',...
    2*ones(Nm-1,1),'b', 0.1*ones(Nm+1,1));

ops_system.brch_fact=[5 5 5 4]; % branching factor of the tree
ops_system.uncertainty='additive'; % kind of uncertainty; 0 for additive; 1 for additive
...and parametric
ops_system.N=10; % prediction horizon

SysMat=SysMat(ops_masses,ops_system);
sys_actual=SysMat.sys;

SysMat.sys=scale_constraints_system(SysMat);

% simple precondition 
SysMat_FBE_smp=SysMat;
SysMat_FBE_smp.sys_ops.precondition='simple';
SysMat_FBE_smp=SysMat_FBE_smp.Precondition_system();

% test points
test_points=15;
%% Algorithm classes- FBE
AlgoFBE_options.ops_FBE.steps=40;
AlgoFBE_options.ops_FBE.epsilon=1e-6;
AlgoFBE_options.algorithm='FBE';
AlgoFBE_options.ops_FBE.memory=5;
%AlgoFBE_options.ops_FBE.lambda=0.5;
AlgoFBE_options.ops_FBE.primal_inf=1e-2;
%AlgoFBE_opgrad_dual_enveloptions.ops_FBE.LS='GOLDSTEIN';

% FBE algorithm-precondition simple 
FBE_algo_smp=Algorithm(SysMat_FBE_smp,AlgoFBE_options);
% Factor step
FBE_algo_smp=FBE_algo_smp.Factor_step();

%% Gurobi algorithm-model

params.outputflag=0;

tic
model=gurobi_solve(sys_actual,SysMat.V,SysMat.tree);
time_gurobi=toc;


%% Algorithm testing
Xtest=4*rand(SysMat.sys.nx,test_points)-2;
iterates=zeros(test_points,1);
grad_steps=zeros(test_points,1);
mean_inner_iter=zeros(test_points,3);
inner_final_iter=zeros(test_points,3);
%seper_value=zeros(2,test_points);
cost_function=zeros(test_points,2);
primal_epsilon=zeros(test_points,2);

%%
for i=1:test_points
    ops_FBE.x0=Xtest(:,i);
    i
    %%
    % FBE Algorithm 
    %{
    FBE_algo_prcnd.algo_details.ops_FBE.lambda=0.1;
    FBE_algo_smp.algo_details.ops_FBE.prox_LS='yes';
    
    [Zclass_LBFS,Yclass_LBFS,Details_LBFS]=FBE_algo_smp.Dual_FBE(ops_FBE.x0);
    
    iterates(i,1)=Details_LBFS.iter;
    if(Details_LBFS.iter>1)
        inner_final_iter(i,1)=Details_LBFS.inner_loops(end);
        if(inner_final_iter(i,1)==10)
            disp('max iterations reached')
            Details_LBFS.tau
        end
        mean_inner_iter(i,1)=mean(Details_LBFS.inner_loops);
    else
        mean_inner_iter(i,1)=0;
    end
    %}
    % Global FBE
    %[Zclass_GlobLBFS,Yclass_GlobLBFS,Details_GlobLBFS]=FBE_algo_smp.Dual_GlobalFBE(ops_FBE.x0);
     [Zclass_GlobLBFS,Yclass_GlobLBFS,Details_GlobLBFS]=FBE_algo_smp.Dual_GlobalFBE_version2(ops_FBE.x0);
    iterates(i,2)=Details_GlobLBFS.iter;
    if(Details_GlobLBFS.iter>1)
        inner_final_iter(i,2)=Details_GlobLBFS.inner_loops(end);
        if(inner_final_iter(i,2)==10)
            disp('max iterations reached')
            Details_GlobLBFS.tau
        end
        mean_inner_iter(i,2)=mean(Details_GlobLBFS.inner_loops);
    else
        mean_inner_iter(i,2)=0;
    end
    %{
    % Accelerated FBE
    [Zclass_AcceLBFS,Yclass_AcceLBFS,Details_AcceLBFS]=...
        FBE_algo_smp.Dual_AccelGlobFBE(ops_FBE.x0);
    %plot(Details_FBE_prcnd.cost_function)
     iterates(i,3)=Details_AcceLBFS.iter;
    if(Details_AcceLBFS.iter>1)
        inner_final_iter(i,3)=Details_AcceLBFS.inner_loops(end);
        if(inner_final_iter(i,3)==10)
            disp('max iterations reached')
            Details_AcceLBFS.tau
        end
        mean_inner_iter(i,3)=mean(Details_AcceLBFS.inner_loops);
    else
        mean_inner_iter(i,3)=0;
    end
    %}
    
    %}
    %  Gurobi
    model.rhs(end-sys_actual.nx+1:end)=ops_FBE.x0;
    tic
    results=gurobi(model,params);
    time_solver(1,i)=toc;
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
    [cost_function(i,1),primal_epsilon(i,1)]=SysMat.cost_function(Zgurobi);
    %[cost_function(i,2),primal_epsilon(i,2)]=SysMat.cost_function(Zclass_LBFS);
    [cost_function(i,3),primal_epsilon(i,3)]=SysMat.cost_function(Zclass_GlobLBFS);
    %[cost_function(i,4),primal_epsilon(i,4)]=SysMat.cost_function(Zclass_AcceLBFS);
end
%%

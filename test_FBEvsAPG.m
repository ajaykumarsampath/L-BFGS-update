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
Nm=5; % Number of masses
T_sampling=0.5;
ops_masses=struct('Nm',Nm,'Ts',T_sampling,'xmin', ...
    -4*ones(2*Nm,1), 'xmax', 4*ones(2*Nm,1), 'umin', -2*ones(Nm-1,1),'umax',...
    2*ones(Nm-1,1),'b', 0.1*ones(Nm+1,1),'M', 4*ones(Nm,1));

ops_system.brch_fact=2*ones(1,10); % branching factor of the tree
ops_system.uncertainty='additive'; % kind of uncertainty; 0 for additive; 1 for additive
...and parametric
ops_system.N=11; % prediction horizon

SysMat=SysMat(ops_masses,ops_system);
sys_actual=SysMat.sys;

SysMat.sys=scale_constraints_system(SysMat);

% FBE-simple
SysMat_FBE_prcnd=SysMat;
SysMat_FBE_prcnd.sys_ops.precondition='simple';
%SysMat_FBE_prcnd.sys_ops.precondition='no';
SysMat_FBE_prcnd=SysMat_FBE_prcnd.Precondition_system();

% APG-simple 
SysMat_APG=SysMat;
SysMat_APG.sys_ops.precondition='simple';
%SysMat_APG.sys_ops.precondition='no';
SysMat_APG=SysMat_APG.Precondition_system();


% FBE-No preconditioning
SysMat_No_Prcnd=SysMat;
SysMat_No_Prcnd.sys_ops.precondition='no';

% APG- No preconditioning
SysMat_APG_NoPrcnd=SysMat;
SysMat_APG_NoPrcnd.sys_ops.precondition='no';
% test points
test_points=10;
%% Algorithm classes- FBE
AlgoFBE_options.ops_FBE.steps=350;
AlgoFBE_options.ops_FBE.epsilon=1e-6;
AlgoFBE_options.ops_FBE.primal_inf=5e-4;
AlgoFBE_options.algorithm='FBE';
AlgoFBE_options.ops_FBE.memory=10;
%AlgoFBE_options.ops_FBE.lambda=0.1;
%AlgoFBE_opgrad_dual_enveloptions.ops_FBE.LS='GOLDSTEIN';

% FBE algorithm
FBE_algo_prcnd=Algorithm(SysMat_FBE_prcnd,AlgoFBE_options);
% Factor step
FBE_algo_prcnd=FBE_algo_prcnd.Factor_step();

% FBE algorithm with 5 memory
AlgoFBE_options.ops_FBE.memory=5;
FBE_algo_mem5=Algorithm(SysMat_FBE_prcnd,AlgoFBE_options);
% Factor step
FBE_algo_mem5=FBE_algo_mem5.Factor_step();


% APG algorithm
AlgoAPG_options.ops_APG.steps=750;
%AlgoAPG_options.ops_APG.lambda=0.1;
AlgoAPG_options.ops_APG.primal_inf=5e-4;
AlgoAPG_options.ops_APG.dual_gap=1e-3;
%AlgoAPG_options.ops_APG.LS='yes';
APG_algo=Algorithm(SysMat_APG,AlgoAPG_options);
APG_algo=APG_algo.Factor_step();

%{
% FBE algorithm- without preconditing
AlgoFBE_options.ops_FBE.lambda=1e-10;
FBE_algo_No_prcnd=Algorithm(SysMat_No_Prcnd,AlgoFBE_options);
FBE_algo_No_prcnd=FBE_algo_No_prcnd.Factor_step();

% APG algorithm- without preconditing
AlgoAPG_options.ops_APG.lambda=0.1;
APG_algo_No_prcnd=Algorithm(SysMat_APG_NoPrcnd,AlgoAPG_options);
APG_algo_No_prcnd=APG_algo_No_prcnd.Factor_step();
%}
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
%%
for i=1:test_points
    ops_FBE.x0=Xtest(:,i);
    %%
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
    
    if(strcmp(results.status,'OPTIMAL'))
        % FBE preconditioned
        %FBE_algo_prcnd.algo_details.ops_FBE.lambda=0.1;
        FBE_algo_prcnd.algo_details.ops_FBE.prox_LS='yes';
        
        [Zclass_FBE_prcnd,Yclass_FBE_prcnd,Details_FBE_prcnd]=FBE_algo_prcnd.Dual_FBEAdaptive(ops_FBE.x0);
        
        iterates(i,1)=Details_FBE_prcnd.iter;
        
        %{
        % FBE No preconditioned
        FBE_algo_No_prcnd.algo_details.ops_FBE.prox_LS='yes';
        
        [Zclass_FBE_No_prcnd,Yclass_FBE_No_prcnd,Details_FBE_No_prcnd]=...
            FBE_algo_No_prcnd.Dual_FBEAdaptive(ops_FBE.x0);
        
        iterates(i,3)=Details_FBE_No_prcnd.iter;
        %}
        %lambda(i,1)=Details_FBE_prcnd.lambda_prox(Details_FBE_prcnd.iter);
        
        % FBE algorithm with memory 5
        FBE_algo_mem5.algo_details.ops_FBE.prox_LS='yes';
        
        [Zclass_FBE_mem5,Yclass_FBE_mem5,Details_FBE_mem5]=...
            FBE_algo_mem5.Dual_FBEAdaptive(ops_FBE.x0);
        
        iterates(i,3)=Details_FBE_mem5.iter;
        
        % APG algorithm with line search
        APG_algo.algo_details.ops_APG.LS='yes';
        [Zclass_APG_LS,Yclass_APG_LS,Details_APG_LS]=APG_algo.Dual_APG(ops_FBE.x0);
        if(isfield(Details_APG_LS,'iterate'))
            iterates(i,2)=Details_APG_LS.iterate;
        else
            iterates(i,2)=750;
        end
        
        %{
        % APG no precondition algorithm with line search
        APG_algo_No_prcnd.algo_details.ops_APG.LS='yes';
        [Zclass_APG_No_LS,Yclass_APG_No_LS,Details_APG_No_LS]=APG_algo_No_prcnd.Dual_APG(ops_FBE.x0);
        if(isfield(Details_APG_No_LS,'iterate'))
            iterates(i,4)=Details_APG_No_LS.iterate;
        else
            iterates(i,4)=250;
        end
        %}
        
    else
        Zclass_APG_LS=Zgurobi;
        Zclass_FBE_mem5=Zgurobi;
        Zclass_FBE_prcnd=Zgurobi;
    end
    
    
    %   
    [cost_function(i,1),primal_epsilon(i,1)]=SysMat.cost_function(Zgurobi);
    [cost_function(i,2),primal_epsilon(i,2)]=SysMat.cost_function(Zclass_FBE_prcnd);
    [cost_function(i,3),primal_epsilon(i,3)]=SysMat.cost_function(Zclass_APG_LS);
    [cost_function(i,4),primal_epsilon(i,4)]=SysMat.cost_function(Zclass_FBE_mem5);
    
    %[cost_function(i,4),primal_epsilon(i,4)]=SysMat.cost_function(Zclass_FBE);
end
%%
figure
semilogy(Details_APG_LS.Glamdba,'LineWidth',1.75);
hold all;
semilogy(Details_FBE_prcnd.Glambda,'LineWidth',1.75);
grid on
%hold all;
semilogy(Details_FBE_mem5.Glambda,'LineWidth',1.75);
axis tight
legend('Dual-APG','LBFGS-Dual-FBE mem 10','LBFGS-Dual-FBE mem 5')
xlabel('Iterations')
ylabel('$\|R_{\lambda}\|^2$','Interpreter','Latex')
%semilogy(Details_APG_No_LS.Glamdba);

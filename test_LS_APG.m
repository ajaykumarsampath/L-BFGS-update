%%
% This script solves optimisation of the scenario-tree based model
% predictive control. The dynamic system considered is the standard
% spring-masss system. This function tests the line search method for 
% calculating the Lipschitz constant
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

% simple-Proximal gradient method
SysMat_APG=SysMat;
SysMat_APG.sys_ops.precondition='Jacobi';
SysMat_APG=SysMat_APG.Precondition_system();

% test points
test_points=5;
%% Algorithm classe
% Algorithm-APG 
AlgoAPG_options.ops_APG.steps=250;
AlgoAPG_options.ops_APG.lambda=0.1;
%AlgoAPG_options.ops_APG.LS='yes';
APG_algo=Algorithm(SysMat_APG,AlgoAPG_options);
APG_algo=APG_algo.Factor_step();

%% Gurobi algorithm-mode
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
    ops_APG.x0=Xtest(:,i);
    %%
    % APG algorithm
    APG_algo.algo_details.ops_APG.LS='no';
    [Zclass_APG,Yclass_APG,Details_APG]=APG_algo.Dual_APG(ops_APG.x0);
    %
    if(isfield(Details_APG,'iterate'))
        iterates(i,2)=Details_APG.iterate;
    else
        iterates(i,2)=250;
    end
    % APG algorithm with line search 
    APG_algo.algo_details.ops_APG.LS='yes';
    [Zclass_APG_LS,Yclass_APG_LS,Details_APG_LS]=APG_algo.Dual_APG(ops_APG.x0);
    if(isfield(Details_APG_LS,'iterate'))
        iterates(i,3)=Details_APG_LS.iterate;
    else
        iterates(i,3)=250;
    end
    %}
    %  Gurobi
    model.rhs(end-sys_actual.nx+1:end)=ops_APG.x0;
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
    
    %%
    
    
    [cost_function(i,2),primal_epsilon(i,2)]=SysMat.cost_function(Zgurobi);
    
    [cost_function(i,4),primal_epsilon(i,4)]=SysMat.cost_function(Zclass_APG);
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
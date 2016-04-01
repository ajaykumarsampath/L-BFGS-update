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
    2*ones(Nm-1,1),'b', 0.1*ones(Nm+1,1));

ops_system.brch_fact=[1 1]; % branching factor of the tree
ops_system.uncertainty='additive'; % kind of uncertainty; 0 for additive; 1 for additive
...and parametric
    ops_system.N=10; % prediction horizon

SysMat=SysMat(ops_masses,ops_system);
sys_actual=SysMat.sys;

SysMat.sys=scale_constraints_system(SysMat);

SysMat_initial=SysMat;
SysMat_initial.sys_ops.constraints='Normalised';

SysMat_initial.sys_ops.precondition='No';
%% Factor step


Algo_options.ops_FBS.steps=1000;
Algo_options.ops_FBS.epsilon=1e-6;
Algo_options.algorithm='FBS';
Algo_options.ops_FBS.memory=5;

APG_initial=Algorithm(SysMat_initial,Algo_options);
APG_initial=APG_initial.Factor_step();


%% Algorithm testing


ops_APG.x0=4*rand(SysMat.sys.nx,1)-2;
%%
[Zclass_init,Yclass_init,Details_init]=APG_initial.Dual_FBS(ops_APG.x0);

Details_init
plot(Details_init.direction)
%% Gurobi algorithm

params.outputflag=0;

tic
model=gurobi_solve(sys_actual,SysMat.V,SysMat.tree);
time_gurobi=toc;

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
    for i=1:non_leaf
        Zgurobi.X(:,i)=results.x((i-1)*nz+1:(i-1)*nz+nx);
        Zgurobi.U(:,i)=results.x((i-1)*nz+nx+1:i*nz);
    end
    for i=1:Ns
        Zgurobi.X(:,non_leaf+i)=results.x(non_leaf*nz+(i-1)*nx+1:non_leaf*nz+i*nx);
    end
else
    disp('Error');
end

%%
% iterations
%plot(iterate);

% plot states
plot(Zclass_init.X(3,:))
hold all;
plot(Zgurobi.X(3,:))

%
norm(Zclass_init.U-Zgurobi.U,inf)
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


ops_system.brch_fact=[2 2]; % branching factor of the tree
ops_system.uncertainty='additive'; % kind of uncertainty; 0 for additive; 1 for additive
...and parametric
    ops_system.N=10; % prediction horizon

SysMat=SysMat(ops_masses,ops_system);

SysMat.sys=scale_constraints_system(SysMat);


SysMat_initial=SysMat;
SysMat_simple_precond=SysMat;
sys_actual=SysMat.sys;


% Jacobi
SysMat.sys_ops.constraints='Normalised';

SysMat.sys_ops.precondition='Jacobi';

SysMat=SysMat.Precondition_system();
%SysMat.sys=scale_constraints_system(SysMat);

% simple
SysMat_simple_precond.sys_ops.precondition='simple';
SysMat_simple_precond=SysMat_simple_precond.Precondition_system();
%SysMat_simple_precond.sys=scale_constraints_system(SysMat_simple_precond);

SysMat_initial.sys_ops.precondition='No';
%% Factor step
APG_precond=Algorithm(SysMat);
APG_precond=APG_precond.Factor_step();


APG_initial=Algorithm(SysMat_initial);
APG_initial=APG_initial.Factor_step();

APG_simple_prcnd=Algorithm(SysMat_simple_precond);
APG_simple_prcnd=APG_simple_prcnd.Factor_step();

%% Algorithm testing

iterate=zeros(20,3);
for i=1:20
    ops_APG.x0=4*rand(SysMat.sys.nx,1)-2;
    
    [Zclass,Yclass,Dclass]=APG_precond.Dual_APG(ops_APG.x0);
    
    [Zclass_init,Yclass_init,Dclass_init]=APG_initial.Dual_APG(ops_APG.x0);
    
    [Zclass_simple,Yclass_simple,Dclass_simple]=APG_simple_prcnd.Dual_APG(ops_APG.x0);
    
    %
    if(isfield(Dclass,'iterate'))
        iterate(i,1)=Dclass.iterate;
    else
        iterate(i,1)=2000;
    end
    
    if(isfield(Dclass_init,'iterate'))
        iterate(i,2)=Dclass_init.iterate;
    else
        iterate(i,2)=2000;
    end
    
    if(isfield(Dclass_simple,'iterate'))
        iterate(i,3)=Dclass_simple.iterate;
    else
        iterate(i,3)=2000;
    end
end


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
plot(iterate);

% plot states
plot(Zclass.X(3,:))
hold all;
%plot(Zclass_init.X(3,:))
plot(Zclass_simple.X(3,:))
plot(Zgurobi.X(3,:))

%%
norm(Zclass.U-Zgurobi.U,inf)
norm(Zclass_simple.U-Zgurobi.U,inf)
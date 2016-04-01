%%
% This function generates a system with different terminal functions and constraints but
% with same size. The constraint are preconditioned accodingly.
% We solve the method using different methods. First formulated using
% 1) Gurobi-IP 2) Gurobi-AS directly

clear all;
close all;
clear model;
clc;
Nm=5; % Number of masses
T_sampling=0.5;
ops_masses=struct('Ts',T_sampling,'xmin', ...
    -4*ones(2*Nm,1), 'xmax', 4*ones(2*Nm,1), 'umin', -2*ones(Nm-1,1),'umax',...
    2*ones(Nm-1,1),'b', 0.1*ones(Nm+1,1));
ops_system.nx=2*Nm;
ops_system.nu=Nm-1;
ops_system.sys_uncert=0;
ops_system.ops_masses=ops_masses;
%sys_no_precond=system_masses(Nm,ops_masses);
various_predict_horz=10;%prediction horizon
Test_points=1;
x_rand=4*rand(ops_system.nx,Test_points)-2;
time_gpad=cell(Test_points,1);
U_max=zeros(2,Test_points);
U_min=zeros(2,Test_points);
%dual_gap=zeros(2,Test_points);
test_cuda=0;

time_solver=zeros(2,Test_points);
result.u=cell(1,1);

params.outputflag=0;
%% Generation of tree
baranching_factor=[2 2 2 2];
for N_prb_steps=4:length(baranching_factor)
    for no_of_pred=1:length(various_predict_horz)
        %ops_system.Np=various_predict_horz(no_of_pred);
        ops.N=various_predict_horz(no_of_pred);
        ops.brch_ftr=ones(ops.N,1);
        ops.brch_ftr(1:N_prb_steps)=baranching_factor(1:N_prb_steps);
        Ns=prod(ops.brch_ftr);
        ops.nx=ops_system.nx;
        ops.prob=cell(ops.N,1);
        for i=1:ops.N;
            if(i<=N_prb_steps)
                pd=rand(1,ops.brch_ftr(i));
                if(i==1)
                    ops.prob{i,1}=pd/sum(pd);
                    pm=1;
                else
                    pm=pm*baranching_factor(i-1);
                    ops.prob{i,1}=kron(ones(pm,1),pd/sum(pd));
                end
            else
                ops.prob{i,1}=ones(Ns,1);
            end
        end
        tic
        [sys_no_precond,Tree]=tree_generation_multiple(ops_system,ops);
        time.tree_formation=toc;
        sys_no_precond.nx=ops_system.nx;
        sys_no_precond.nu=ops_system.nu;
        sys_no_precond.Np=ops.N;

        %%
        %Cost functioin
        V.Q=eye(sys_no_precond.nx);
        V.R=eye(sys_no_precond.nu);
        %%terminal constraints
        sys_no_precond.Ft=cell(Ns,1);
        sys_no_precond.gt=cell(Ns,1);
        V.Vf=cell(Ns,1);
        sys_no_precond.trm_size=(2*sys_no_precond.nx)*ones(Ns,1);
        %r=rand(Ns,1);
        r=ones(Ns,1);
        for i=1:Ns
            %consitraint in the horizon
            sys_no_precond.Ft{i}=[eye(sys_no_precond.nx);-eye(sys_no_precond.nx)];
            sys_no_precond.gt{i}=(3+0.1*rand(1))*ones(2*sys_no_precond.nx,1);
            nt=size(sys_no_precond.Ft{i},1);
            P=Polyhedron('A',sys_no_precond.Ft{i},'b',sys_no_precond.gt{i});
            if(isempty(P))
                error('Polyhedron is empty');
            end
            V.Vf{i}=dare(sys_no_precond.A{1},sys_no_precond.B{1},r(i)*V.Q,r(i)*V.R);
            %V.Vf{i}=dare(sys_no_precond.A,sys_no_precond.B,r(i)*V.Q,r(i)*V.R);
        end
        %normalizing constraints
        sys_no_precond=Normalise_constraints(sys_no_precond);
        %% preconditioning the system and solve the system using dgpad.
        
        [sys,Hessian_iapp]=calculate_diffnt_precondition_matrix(sys_no_precond,V,Tree...
            ,struct('use_cell',1,'use_hessian',0));
        tic;
        Ptree=APG_factor_step(sys,V,Tree);
        toc
        
        ops_GPAD.steps=2000;
        ops_GPAD.primal_inf=1e-3;
        ops_GPAD.dual_gap=1e-2;
        ops_GPAD.alpha=1/calculate_Lipschitz(sys,V,Tree);
        
        tic
        model=gurobi_solve(sys_no_precond,V,Tree);
        time_gurobi=toc;

        max_size=zeros(Test_points,length(Tree.stage));
        for kk=1:Test_points
            
            ops_GPAD.x0=x_rand(:,kk);
            
            %GPAD
            if(kk==1)
            [Z_gpad_pre,Y_gpad_pre,time_gpad{kk}]=Dual_APG_algorithm(sys,Ptree,Tree,V,ops_GPAD);
            if(~isfield(time_gpad{kk},'iterate'))
                time_gpad{kk}.iterate=ops_GPAD.steps;
            end
            end
            %  Gurobi
            model.rhs(end-sys_no_precond.nx+1:end)=ops_GPAD.x0;
            tic
            results=gurobi(model,params);
            time_solver(1,kk)=toc;
            if(strcmp(results.status,'OPTIMAL'))
                disp('OK');
            else
                disp('Error');
            end
        end
    end
end

%% testing the dual gradient calculation
figure(1)
plot(time_gpad{1}.prm_cst);
hold all;
plot(time_gpad{1}.dual_cst);
xlabel('iterates')
ylabel('cost')
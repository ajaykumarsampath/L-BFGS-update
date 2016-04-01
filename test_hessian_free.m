%%

clear all;
close all;
clear model;
clc;
Nm=2; % Number of masses
T_sampling=0.5;
ops_masses=struct('Ts',T_sampling,'xmin', ...
    -4*ones(2*Nm,1), 'xmax', 4*ones(2*Nm,1), 'umin', -2*ones(Nm-1,1),'umax',...
    2*ones(Nm-1,1),'b', 0.1*ones(Nm+1,1));
ops_system.nx=2*Nm;
ops_system.nu=Nm-1;
ops_system.sys_uncert=0;
ops_system.ops_masses=ops_masses;
%sys_no_precond=system_masses(Nm,ops_masses);
predict_horz=5;%prediction horizon
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
branch_factor=[1];
%ops_system.Np=various_predict_horz(no_of_pred);
ops.N=predict_horz; %step 2: argmin of the lagrangian using dynamic programming
ops.brch_ftr=ones(ops.N,1);
ops.brch_ftr(1:size(branch_factor,2))=branch_factor;
Ns=prod(ops.brch_ftr);
ops.nx=ops_system.nx;
ops.prob=cell(ops.N,1);
for i=1:ops.N;
    if(i<=length(branch_factor))
        pd=rand(1,ops.brch_ftr(i));
        if(i==1)
            ops.prob{i,1}=pd/sum(pd);
            pm=1;
        else
            pm=pm*branch_factor(i-1);
            ops.prob{i,1}=kron(ones(pm,1),pd/sum(pd));
        end
    else
        ops.prob{i,1}=ones(Ns,1);
    end
end
tic
[sys,Tree]=tree_generation_multiple(ops_system,ops);
time.tree_formation=toc;
sys.nx=ops_system.nx;
sys.nu=ops_system.nu;
sys.Np=ops.N;
Tree.value=10*Tree.value;

%%
%Cost functioin
V.Q=eye(sys.nx);
V.R=eye(sys.nu);
%%terminal constraints
sys.Ft=cell(Ns,1);
sys.gt=cell(Ns,1);
V.Vf=cell(Ns,1);
sys.trm_size=(2*sys.nx)*ones(Ns,1);
%r=rand(Ns,1);
r=ones(Ns,1);
for i=1:Ns
    %consitraint in the horizon
    sys.Ft{i}=[eye(sys.nx);-eye(sys.nx)];
    sys.gt{i}=(3+0.1*rand(1))*ones(2*sys.nx,1);
    nt=size(sys.Ft{i},1);
    P=Polyhedron('A',sys.Ft{i},'b',sys.gt{i});
    if(isempty(P))
        error('Polyhedron is empty');
    end
    V.Vf{i}=dare(sys.A{1},sys.B{1},r(i)*V.Q,r(i)*V.R);
end

% Normalizing constraints
sys=Normalise_constraints(sys);
[sys,Hessian_iapp]=calculate_diffnt_precondition_matrix(sys,V,Tree...
    ,struct('use_cell',1,'use_hessian',0));

%% Caculation of dual gradient 
tic;
Ptree=APG_factor_step(sys,V,Tree);
toc

Ns=length(Tree.leaves);% total scenarios in the tree
Nd=length(Tree.stage);%toal nodes in the tree
non_leaf=Nd-Ns;
epsilon=1e-6;
nz=sys.nx+sys.nu;
ny=2*nz;
% Initalizing the dual varibables
Y0.y=1000*rand(non_leaf,size(sys.F{1},1));
Y1.y=200*rand(non_leaf,size(sys.F{1},1));
for i=1:Ns
    Y0.yt{i,:}=zeros(1,size(sys.Ft{i,1},1));
    Y1.yt{i,:}=zeros(1,size(sys.Ft{i,1},1));
end

ops.x0=4*rand(ops_system.nx,Test_points)-2;
Z1=APG_solve_step(sys,Ptree,Tree,Y0,ops.x0);

W.y=Y0.y+epsilon*Y1.y;
W.yt=Y0.yt;

vector_y1=zeros(non_leaf*ny+Ns*2*sys.nx,1);
vector_y2=zeros(non_leaf*ny+Ns*2*sys.nx,1);

for i=1:non_leaf
    vector_y1((i-1)*ny+1:i*ny,1)=Y0.y(i,:)';
    vector_y2((i-1)*ny+1:i*ny,1)=Y1.y(i,:)';
end

for i=1:Ns
    vector_y1(non_leaf*ny+2*(i-1)*sys.nx+1:non_leaf*ny+2*i*sys.nx)=Y0.yt{i};
    vector_y2(non_leaf*ny+2*(i-1)*sys.nx+1:non_leaf*ny+2*i*sys.nx)=Y1.yt{i};
end
vector_z=vector_y1+epsilon*vector_y2;

Z2=APG_solve_step(sys,Ptree,Tree,W,ops.x0);


if(epsilon>0)
    Hess_approx.X=(Z2.X-Z1.X)/epsilon;
    Hess_approx.U=(Z2.U-Z1.U)/epsilon;
else
    Hess_approx.X=(Z2.X-Z1.X);
    Hess_approx.U=(Z2.U-Z1.U);
end

vect_approx.y=zeros(ops.N,ny);
vect_approx.yt{1}=zeros(1,2*sys.nx);

for i=1:non_leaf
    vect_approx.y(i,:)=-([sys.F{i} sys.G{i}]*[Hess_approx.X(:,i);Hess_approx.U(:,i)])';
end 


%% actual hessian calculation 
% Build the KKT condition matrix

row_KKT=non_leaf*nz+Ns*sys.nx+(non_leaf+1)*sys.nx;
col_KKT=Nd*sys.nx+(non_leaf)*sys.nu;
off_set=non_leaf*nz+Ns*sys.nx;
KKT=zeros(row_KKT,row_KKT);
KKT_equality=zeros(non_leaf*sys.nx,col_KKT);

%KKT(1:non_leaf*nz,1:non_leaf*nz)=kron(blkdiag(V.Q,V.R),ones(non_leaf,1));
KKT(1:non_leaf*nz,1:non_leaf*nz)=kron(diag(Tree.prob(1:non_leaf)),blkdiag(2*V.Q,2*V.R));
for i=1:Ns
    KKT(non_leaf*nz+(i-1)*sys.nx+1:non_leaf*nz+i*sys.nx,...
        non_leaf*nz+(i-1)*sys.nx+1:non_leaf*nz+i*sys.nx)=2*V.Vf{i};
end

for i=1:non_leaf
    KKT_equality((i-1)*sys.nx+1:i*sys.nx,(i-1)*nz+1:i*nz)=[sys.A{i} sys.B{i}];
    for j=1:length(Tree.children{i})
        KKT_equality((i-1)*sys.nx+1:i*sys.nx,(Tree.children{i}(j)-1)*nz+1:...
            (Tree.children{i}(j)-1)*nz+sys.nx)=-eye(sys.nx);
    end
end
KKT_equality(non_leaf*sys.nx+1:(non_leaf+1)*sys.nx,1:sys.nx)=eye(sys.nx);
KKT(col_KKT+1:row_KKT,1:col_KKT)=KKT_equality;
KKT(1:col_KKT,col_KKT+1:row_KKT)=KKT_equality';
inv_KKT=KKT\eye(row_KKT);

K11=inv_KKT(1:col_KKT,1:col_KKT);
K12=inv_KKT(1:col_KKT,col_KKT+1:row_KKT);
M=0.5*KKT(1:off_set,1:off_set);
dual_quad=-2*(K11'*M*K11-K11);
%dual_quad=K11;
%
RHS_K11=zeros(row_KKT,1);
for i=1:non_leaf
    RHS_K11((i-1)*nz+1:i*nz,1)=-[sys.F{i} sys.G{i}]'*Y0.y(i,:)';
end 
for i=1:Ns
    RHS_K11(non_leaf*nz+(i-1)*sys.nx+1:non_leaf*nz+i*sys.nx,1)=-[sys.Ft{i}]'*Y0.yt{i}';
end

RHS_K11_z=zeros(row_KKT,1);
for i=1:non_leaf
    RHS_K11_z((i-1)*nz+1:i*nz,1)=-[sys.F{i} sys.G{i}]'*W.y(i,:)';
end 
for i=1:Ns
    RHS_K11_z(non_leaf*nz+(i-1)*sys.nx+1:non_leaf*nz+i*sys.nx,1)=-[sys.Ft{i}]'*W.yt{i}';
end


for i=1:non_leaf
    for j=1:length(Tree.children{i})
    RHS_K11(off_set+(Tree.children{i}(j)-2)*sys.nx+1:...
        off_set+(Tree.children{i}(j)-1)*sys.nx,1)=-Tree.value(Tree.children{i}(j),:)';
    end
end 
RHS_K11(row_KKT-sys.nx+1:row_KKT,1)=Z1.X(:,1);

RHS_K11_z(off_set+1:end,1)=RHS_K11(off_set+1:end,1);


LHS_K11=inv_KKT*RHS_K11;
LHS_K11_z=inv_KKT*RHS_K11_z;

Z_actual.X=zeros(sys.nx,Nd);
Z_actual.U=zeros(sys.nu,non_leaf);
Z_actual.X(:,1)=ops.x0;

for i=1:non_leaf
    Z_actual.X(:,i)=LHS_K11((i-1)*nz+1:(i-1)*nz+sys.nx,1);
    Z_actual.U(:,i)=LHS_K11((i-1)*nz+sys.nx+1:i*nz,1);
end 

Z_actual_z.X=zeros(sys.nx,Nd);
Z_actual_z.U=zeros(sys.nu,non_leaf);
Z_actual_z.X(:,1)=ops.x0;

for i=1:non_leaf
    Z_actual_z.X(:,i)=LHS_K11_z((i-1)*nz+1:(i-1)*nz+sys.nx,1);
    Z_actual_z.U(:,i)=LHS_K11_z((i-1)*nz+sys.nx+1:i*nz,1);
end 

temp_grad=K11*RHS_K11(1:off_set,1)+K12*RHS_K11(off_set+1:end,1);
temp_grad_z=K11*RHS_K11_z(1:off_set,1)+K12*RHS_K11_z(off_set+1:end,1);

for i=1:non_leaf
    Z_actual1.X(:,i)=temp_grad((i-1)*nz+1:(i-1)*nz+sys.nx,1);
    Z_actual1.U(:,i)=temp_grad((i-1)*nz+sys.nx+1:i*nz,1);
    
    Z_actual1_z.X(:,i)=temp_grad_z((i-1)*nz+1:(i-1)*nz+sys.nx,1);
    Z_actual1_z.U(:,i)=temp_grad_z((i-1)*nz+sys.nx+1:i*nz,1);
end 

dual_hessian=zeros((Nd-Ns)*ny+Ns*2*sys.nx,(Nd-Ns)*ny+Ns*2*sys.nx);

for i=1:Nd-Ns
    dual_hessian((i-1)*ny+1:i*ny,(i-1)*ny+1:i*ny)=[sys.F{i} sys.G{i}]*...
        dual_quad((i-1)*nz+1:i*nz,(i-1)*nz+1:i*nz)*[sys.F{i} sys.G{i}]';
end

for i=1:Ns
    dual_hessian(non_leaf*ny+2*(i-1)*sys.nx+1:non_leaf*ny+2*i*sys.nx,...
        non_leaf*ny+2*(i-1)*sys.nx+1:non_leaf*ny+2*i*sys.nx)=sys.Ft{i}*...
        dual_quad(non_leaf*nz+(i-1)*sys.nx+1:non_leaf*nz+i*sys.nx,non_leaf*nz+(i-1)*sys.nx+1....
        :non_leaf*nz+i*sys.nx)*sys.Ft{i}';
end


%%

if(epsilon>0)
    Hess_approx1.X=(Z_actual_z.X-Z_actual.X)/epsilon;
    Hess_approx1.U=(Z_actual_z.U-Z_actual.U)/epsilon;
end
vect_approx1.y=zeros(ops.N,ny);
vect_approx1.yt{1}=zeros(1,2*sys.nx);

for i=1:non_leaf
    vect_approx1.y(i,:)=-([sys.F{i} sys.G{i}]*[Hess_approx1.X(:,i);Hess_approx1.U(:,i)])';
end 

%%

vect_actual.y=zeros(non_leaf,ny);
%vector_actual.U=zeros(sys.nu,non_leaf);

for i=1:non_leaf
    vect_actual.y(i,1:2*sys.nx)=(dual_hessian((i-1)*ny+1:(i-1)*ny+2*sys.nx,...
        (i-1)*ny+1:i*ny)*Y1.y(i,:)')';
    vect_actual.y(i,2*sys.nx+1:ny)=(dual_hessian((i-1)*ny+2*sys.nx+1:i*ny,...
        (i-1)*ny+1:i*ny)*Y1.y(i,:)')';
end


vect_actual2.y=zeros(non_leaf,ny);
temp_var=(dual_hessian*vector_z-dual_hessian*vector_y1)/epsilon;
for i=1:non_leaf
    vect_actual2.y(i,:)=temp_var((i-1)*ny+1:i*ny);
end

vect_actual3.y=zeros(non_leaf,ny);
temp_var=(temp_grad_z-temp_grad)/epsilon;
temp_var1=K11*(RHS_K11_z(1:off_set,1)-RHS_K11(1:off_set,1))/epsilon;
for i=1:non_leaf
    vect_actual3.y(i,:)=(-[sys.F{i} sys.G{i}]*temp_var((i-1)*nz+1:i*nz))';
    %vect_actual3.y(i,:)=-temp_var((i-1)*nz+1:i*nz);
end

vect_actual5.y=zeros(non_leaf,ny);
for i=1:non_leaf
    vect_actual5.y(i,:)=(-[sys.F{i} sys.G{i}]*...
        K11((i-1)*nz+1:i*nz,(i-1)*nz+1:i*nz)*[sys.F{i} sys.G{i}]'*(W.y(i,:)-Y0.y(i,:))')'/epsilon;
    %vect_actual3.y(i,:)=-temp_var((i-1)*nz+1:i*nz);
end


for i=1:non_leaf
    temp_var5((i-1)*nz+1:i*nz,1)=-K11((i-1)*nz+1:i*nz,(i-1)*nz+1:i*nz)*...
        [sys.F{i} sys.G{i}]'*(W.y(i,:)-Y0.y(i,:))'/epsilon;
end

for i=1:non_leaf
    temp_var6((i-1)*nz+1:i*nz,1)=K11((i-1)*nz+1:i*nz,(i-1)*nz+1:i*nz)*...
        (RHS_K11_z((i-1)*nz+1:i*nz,1)-RHS_K11((i-1)*nz+1:i*nz,1))/epsilon;
end

for i=1:Nd-Ns
    dual_hessian_1((i-1)*nz+1:i*nz,(i-1)*ny+1:i*ny)=K11((i-1)*nz+1:i*nz,...
        (i-1)*nz+1:i*nz)*[sys.F{i} sys.G{i}]';
end

for i=1:Ns
    dual_hessian_1(non_leaf*nz+(i-1)*sys.nx+1:non_leaf*nz+i*sys.nx,...
        non_leaf*ny+2*(i-1)*sys.nx+1:non_leaf*ny+2*i*sys.nx)=...
        K11(non_leaf*nz+(i-1)*sys.nx+1:non_leaf*nz+i*sys.nx,non_leaf*nz+(i-1)*sys.nx+1....
        :non_leaf*nz+i*sys.nx)*sys.Ft{i}';
end

temp_var=(dual_hessian_1*vector_z-dual_hessian_1*vector_y1)/epsilon;
for i=1:non_leaf
    vect_actual4.y(i,:)=-[sys.F{i} sys.G{i}]*temp_var((i-1)*nz+1:i*nz);
end

%%
i=2
(RHS_K11_z((i-1)*nz+1:i*nz,1)-RHS_K11((i-1)*nz+1:i*nz,1))'/epsilon

K11((i-1)*nz+1:i*nz,:)*(RHS_K11_z(1:off_set,1)-RHS_K11(1:off_set,1))/epsilon
%%
%{
Z_actual.X=zeros(sys.nx,Nd);
Z_actual.U=zeros(sys.nu,non_leaf);
Z_actual.X(:,1)=ops.x0;

for i=1:non_leaf
    Z_actual.X(:,i)=LHS_K11((i-1)*nz+1:(i-1)*nz+sys.nx,1);
    Z_actual.U(:,i)=LHS_K11((i-1)*nz+sys.nx+1:i*nz,1);
end 

Z_actual_z.X=zeros(sys.nx,Nd);
Z_actual_z.U=zeros(sys.nu,non_leaf);
Z_actual_z.X(:,1)=ops.x0;

for i=1:non_leaf
    Z_actual_z.X(:,i)=LHS_K11_z((i-1)*nz+1:(i-1)*nz+sys.nx,1);
    Z_actual_z.U(:,i)=LHS_K11_z((i-1)*nz+sys.nx+1:i*nz,1);
end 
%}

%{
vect_actual1.y=zeros(non_leaf,ny);
for i=1:non_leaf
    vect_actual1.y(i,1:2*sys.nx)=(dual_hessian((i-1)*ny+1:(i-1)*ny+2*sys.nx,...
        (i-1)*ny+1:i*ny)*vector_y2((i-1)*ny+1:i*ny))';
    vect_actual1.y(i,2*sys.nx+1:ny)=(dual_hessian((i-1)*ny+2*sys.nx+1:i*ny,...
        (i-1)*ny+1:i*ny)*vector_y2((i-1)*ny+1:i*ny))';
end
%}
%% temp_var=dual_hessian*vector_y2;
% vect_actual2.y=zeros(non_leaf,ny);

%{
Z_actual.X=zeros(sys.nx,Nd);
Z_actual.U=zeros(sys.nu,non_leaf);
Z_actual.X(:,1)=ops.x0;

for i=1:non_leaf
    Z_actual.X(:,i)=LHS_K11((i-1)*nz+1:(i-1)*nz+sys.nx,1);
    Z_actual.U(:,i)=LHS_K11((i-1)*nz+sys.nx+1:i*nz,1);
end 


temp_var=K11*RHS_K11(1:off_set,1)+K12*RHS_K11(off_set+1:end,1);
for i=1:non_leaf
    Z_actual1.X(:,i)=temp_var((i-1)*nz+1:(i-1)*nz+sys.nx,1);
    Z_actual1.U(:,i)=temp_var((i-1)*nz+sys.nx+1:i*nz);
end
temp_var3=KKT_equality*temp_var-RHS_K11(off_set+1:end,1);

temp_var1=K11*(KKT(1:off_set,1:off_set)*temp_var-RHS_K11(1:off_set));

%%
inv_M=(M)\eye(off_set,off_set);

AMAtran_mat=KKT_equality*inv_M*KKT_equality';
K11_form2=0.5*(inv_M-inv_M*KKT_equality'*((AMAtran_mat)\KKT_equality)*inv_M);
K12_form2=inv_M*KKT_equality'/(AMAtran_mat);


%%
dual_quad_1=-2*0.25*(inv_M'*KKT_equality'*((AMAtran_mat)\KKT_equality)*inv_M-inv_M);

for i=1:Nd-Ns
    dual_hessian_1((i-1)*ny+1:i*ny,(i-1)*ny+1:i*ny)=[sys.F{i} sys.G{i}]*...
        dual_quad_1((i-1)*nz+1:i*nz,(i-1)*nz+1:i*nz)*[sys.F{i} sys.G{i}]';
end

for i=1:Ns
   dual_hessian_1(non_leaf*ny+2*(i-1)*sys.nx+1:non_leaf*ny+2*i*sys.nx,...
       non_leaf*ny+2*(i-1)*sys.nx+1:non_leaf*ny+2*i*sys.nx)=sys.Ft{i}*...
       dual_quad_1(non_leaf*nz+(i-1)*sys.nx+1:non_leaf*nz+i*sys.nx,non_leaf*nz+(i-1)*sys.nx+1....
       :non_leaf*nz+i*sys.nx)*sys.Ft{i}';
end

inv_M=inv(0.5*KKT(1:non_leaf*nz+Ns*sys.nx,1:non_leaf*nz+Ns*sys.nx));
AMAtran_mat=KKT_equality*inv_M*KKT_equality';
K11_2=0.5*(inv_M-inv_M*KKT_equality'*inv(AMAtran_mat)*KKT_equality*inv_M);
%}
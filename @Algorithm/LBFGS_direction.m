function [ obj,dir_env ] = LBFGS_direction( obj,Grad_env,Grad_envOld,Y,Yold)
%
% This function calculate the direction using quasi-newton method- limited
% memory BFGS method.
%
sys=obj.SysMat_.sys;
Lbfgs=obj.algo_details.ops_FBS.Lbfgs;
memory=obj.algo_details.ops_FBS.memory;
alphaC=obj.algo_details.ops_FBS.alphaC;

nx=sys.nx;
nu=sys.nu;
nz=nx+nu;
ny=2*nz;
Nd=length(obj.SysMat_.tree.stage);
Ns=length(obj.SysMat_.tree.leaves);
non_leaf=Nd-Ns;

Siter=zeros(non_leaf*ny+2*Ns*nx,1);
GradIter=zeros(non_leaf*ny+2*Ns*nx,1);


Siter(1:non_leaf*ny,1)=vec(Y.y-Yold.y);
Siter(non_leaf*ny+1:non_leaf*ny+2*Ns*nx,1)=cell2mat(Y.yt)-cell2mat(Yold.yt);


GradIter(1:non_leaf*ny,1)=vec(Grad_env.y);
GradIter(non_leaf*ny+1:non_leaf*ny+2*Ns*nx,1)=cell2mat(Grad_env.yt);

GradIterOld(1:non_leaf*ny,1)=vec(Grad_envOld.y);
GradIterOld(non_leaf*ny+1:non_leaf*ny+2*Ns*nx,1)=cell2mat(Grad_envOld.yt);

Yk  = GradIter-GradIterOld;
YSk = Siter'*GradIter;

if norm(GradIter) < 1,alphaC = 3;end
if YSk/(Siter'*Siter) > 1e-6*norm(GradIter) ^alphaC
    Lbfgs.LBFGS_col = 1 + mod(Lbfgs.LBFGS_col, memory);
    Lbfgs.LBFGS_mem = min(Lbfgs.LBFGS_mem+1, memory);
    Lbfgs.S(:,Lbfgs.LBFGS_col) = Siter;
    Lbfgs.Y(:,Lbfgs.LBFGS_col) = Yk;
    Lbfgs.YS(Lbfgs.LBFGS_col)  = YSk;
else
    Lbfgs.skipCount = Lbfgs.skipCount+1;
end
Lbfgs.H = YSk/(Yk'*Yk);
Dir_env = LBFGS(Lbfgs.S,Lbfgs.Y,Lbfgs.YS,Lbfgs.H,...
    -GradIter, int32(Lbfgs.LBFGS_col), int32(Lbfgs.LBFGS_mem));

obj.algo_details.ops_FBS.Lbfgs=Lbfgs;
dir_env.y=reshape(Dir_env(1:non_leaf*ny,1),ny,non_leaf);

for i=1:Ns
    dir_env.yt{i,1}=Dir_env(non_leaf*ny+(i-1)*2*nx+1:non_leaf*ny+i*2*nx,1);
end 

end


function [ Lbfgs,dir_env ] = LBFGS_direction_version2( obj,Grad_env,Grad_envOld,Y,Yold)
%
% This function calculate the direction using quasi-newton method- limited
% memory BFGS method.
%
% Syntax : [ obj,dir_env ] = LBFGS_direction( obj,Grad_env,Grad_envOld,Y,Yold)
%
% Input  :    obj          :   Algorithm object 
%             Grad_env     :   Gradient of the envelope
%             Grav_envOld  :   Old gradient envelope 
%             Y            :   Present state 
%             Yold         :   Old state
% 
% Output :    obj          :   Algorithm object 
%             dir_env      :   Direction calculated with LBFGS method
%
%

sys=obj.SysMat_.sys;
Lbfgs=obj.algo_details.ops_FBE.Lbfgs;
memory=obj.algo_details.ops_FBE.memory;
alphaC=obj.algo_details.ops_FBE.alphaC;

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
%Siter=-Siter;
%Yk=-Yk;
YSk = Yk'*Siter;

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
H = YSk/(Yk'*Yk);
if(H<0)
    Lbfgs.H=1;
else
    Lbfgs.H=H;
end
%Lbfgs.H = YSk/(Yk'*Yk);
%Lbfgs.H =1;
Lbfgs.Hnum=YSk;
Lbfgs.Hden=(Yk'*Yk);
Lbfgs.rho=Lbfgs.YS(Lbfgs.LBFGS_col); 

Dir_env = LBFGS(Lbfgs.S,Lbfgs.Y,Lbfgs.YS,Lbfgs.H,...
    -GradIter, int32(Lbfgs.LBFGS_col), int32(Lbfgs.LBFGS_mem));

obj.algo_details.ops_FBE.Lbfgs=Lbfgs;
dir_env.y=reshape(Dir_env(1:non_leaf*ny,1),ny,non_leaf);

for i=1:Ns
    dir_env.yt{i,1}=Dir_env(non_leaf*ny+(i-1)*2*nx+1:non_leaf*ny+i*2*nx,1);
end 

end



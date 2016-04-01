function [ obj ] = Precondition_system(obj)
% 
% Function calculates the precondition matrix 
% 
% Syntax :  [sys]=Precondition_system(SysMat); 
%        
% Input  :  SysMat     :
% 
% Output :  sys        :
%

sys=obj.sys;
tree=obj.tree;
V=obj.V;
if(strcmp(obj.sys_ops.precondition,'Jacobi'))
    % Create the Create the dual-hessian of the system with single scenario
    % 
    Nd=length(tree.stage);
    Ns=length(tree.leaves);
    nx=sys.nx;
    nu=sys.nu;
    nz=nx+nu;
    nc=size(sys.F{1},1);
    nc_t=size(sys.Ft{1},1);
    
    Np=sys.Np;
    
    size_KKT=Np*nz+nx+(Np+1)*nx;
    off_set=Np*nz+nx; 
    
    KKT=zeros(size_KKT,size_KKT);
    
    KKT(1:Np*nz,1:nz*Np)=kron(eye(Np),blkdiag(2*V.Q,2*V.R));
    KKT(Np*nz+1:Np*nz+nx,Np*nz+1:Np*nz+nx)=V.Vf{1};
    
    KKT_equality=zeros((Np+1)*nx,Np*nz+nx);
    
    for i=1:Np
        KKT_equality((i-1)*nx+1:i*nx,(i-1)*nz+1:i*nz+nx)=[sys.A{1} sys.B{1} -eye(nx)];
    end 
    KKT_equality(Np*nx+1:(Np+1)*nx,1:nx)=eye(nx);
    KKT(off_set+1:end,1:Np*nz+nx)=KKT_equality;
    KKT(1:Np*nz+nx,Np*nz+nx+1:Np*nz+nx+(Np+1)*nx)=KKT_equality';
    
    inv_KKT=KKT\eye(size_KKT);
    K11=inv_KKT(1:off_set,1:off_set);
    
    Fsys=zeros(nc*Np+nc_t,Np*nz+nx);
    for i=1:Np
        Fsys((i-1)*nc+1:i*nc,(i-1)*nz+1:i*nz)=[sys.F{1} sys.G{1}];
    end
    Fsys(Np*nc+1:Np*nc+nc_t,Np*nz+1:Np*nz+nx)=sys.Ft{1};
    
    dual_hessian=Fsys*K11*Fsys';
    obj.sys_ops.Lipschitz=1/norm(dual_hessian,2);
    
    diag_dual_hessian=diag(dual_hessian);
    diag_dual_hessian(1:2*nx)=0;
    inv_sqrt_diag_dual_hessian=1./sqrt(diag_dual_hessian);
    inv_sqrt_diag_dual_hessian(1:2*nx)=0;
    for i=1:Nd-Ns
        j=tree.stage(i)+1;
        sys.F{i}=sqrt(tree.prob(i))*diag(inv_sqrt_diag_dual_hessian((j-1)*nc+1:j*nc))...
            *sys.F{i};
        sys.G{i}=sqrt(tree.prob(i))*diag(inv_sqrt_diag_dual_hessian((j-1)*nc+1:j*nc))...
            *sys.G{i};
        sys.g{i}=sqrt(tree.prob(i))*diag(inv_sqrt_diag_dual_hessian((j-1)*nc+1:j*nc))...
            *sys.g{i};
    end
    
    for i=1:Ns
        sys.Ft{i}=sqrt(tree.prob(tree.leaves(i)))*...
            diag(inv_sqrt_diag_dual_hessian(Np*nc+1:end))*sys.Ft{i};
        sys.gt{i}=sqrt(tree.prob(tree.leaves(i)))*...
            diag(inv_sqrt_diag_dual_hessian(Np*nc+1:end))*sys.gt{i};
    end 
else
    Nd=length(tree.stage);
    Ns=length(tree.leaves);
    %H1=sys.F{1}*(V.Q\sys.F{1}')+sys.G{1}*(V.R\sys.G{1}');
    %H1=eye(2*sys.nx+2*sys.nu);
    for i=1:Nd-Ns
        %H=sqrt(diag(diag(H1)));
        sys.F{i}=sqrt(tree.prob(i))*(sys.F{i});
        sys.G{i}=sqrt(tree.prob(i))*(sys.G{i});
        sys.g{i}=sqrt(tree.prob(i))*(sys.g{i});
    end
    %{
    for i=1:Ns
        H1=sys.Ft{i}*(V.Vf{i}\sys.Ft{i}');
        %Ht{i}=sqrt(diag(diag(H1)));
        Ht{i}=eye(2*sys.nx);
    end
    %}
    for i=1:Ns
        sys.Ft{i}=sqrt(tree.prob(tree.leaves(i)))*sys.Ft{i};
        sys.gt{i}=sqrt(tree.prob(tree.leaves(i)))*sys.gt{i};
    end
    
end
obj.sys=sys;

end


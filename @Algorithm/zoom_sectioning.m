function [alpha,details_zoom]=zoom_sectioning(obj,Y,d,aLo,aHo,ops)

tree=obj.SysMat_.tree;
non_leaf=length(tree.children);
Ns=length(tree.leaves);

%sys=obj.SysMat_.sys;
lambda=obj.algo_details.ops_FBE.lambda;
V=obj.SysMat_.V;
phi0=ops.phi0;
x0=ops.x0;
i=1;
details_zoom.term_WF=0;
while(i>0)
    
    alpha_c = 1/2*(aLo+aHo);
    
    % New direction
    Ynew.y=Y.y+alpha_c*d.y;
    for j=1:ops.Ns
        Ynew.yt{j,1}=Y.yt{j,1}+alpha_c*d.yt{j,1};
    end 
    
    [Grad_new,Znew,details_new] =obj.grad_dual_envelop(Ynew,x0);
    lambda_new=details_new.lambda;
    %lambda_new
    separ_var_new.y=details_new.Hx-details_new.T.y;
    for j=1:ops.Ns
       separ_var_new.yt{j,1}=details_new.Hx_term{j,1}-details_new.T.yt{j,1}; 
    end 
    
    
    phi_c=0;
    new_curv_dir=0;
    for j=1:non_leaf
        phi_c=phi_c+tree.prob(j)*Znew.X(:,j)'*V.Q*Znew.X(:,j)+tree.prob(j)*Znew.U(:,j)'...
            *V.R*Znew.U(:,j)+0.5*lambda_new*(separ_var_new.y(:,j)'*...
            separ_var_new.y(:,j))+Ynew.y(:,j)'*separ_var_new.y(:,j);
        new_curv_dir=new_curv_dir+Grad_new.y(:,j)'*d.y(:,j);
    end
    
    for j=1:Ns
        phi_c=phi_c+tree.prob(non_leaf+j)*Znew.X(:,non_leaf+j)'*V.Vf{j}*...
            Znew.X(:,non_leaf+j)+Ynew.yt{j}'*separ_var_new.yt{j,1}+0.5*lambda_new*...
            norm(separ_var_new.yt{j,1})^2;
        new_curv_dir=new_curv_dir+Grad_new.yt{j,1}'*d.yt{j,1};
    end
    phi_c=-phi_c;
    %new_curv_dir=-new_curv_dir;
    
     % Lower bound
    YaLo.y=Y.y+aLo*d.y;
    for j=1:Ns
        YaLo.yt{j,1}=Y.yt{j,1}+aLo*d.yt{j,1};
    end
    
    [Grad_aLo,ZaLo,details_aLo] =obj.grad_dual_envelop(YaLo,x0);
    lambda_Lo=details_aLo.lambda;
    % lambda_Lo
    separ_aLo.y=details_aLo.Hx-details_aLo.T.y;
    for j=1:Ns
        separ_aLo.yt{j,1}=details_aLo.Hx_term{j,1}-details_aLo.T.yt{j,1};
    end 
    
    phiLo=0;
    curv_dir_oLo=0;
    for j=1:non_leaf
        phiLo=phiLo+tree.prob(j)*ZaLo.X(:,j)'*V.Q*ZaLo.X(:,j)+tree.prob(j)*ZaLo.U(:,j)'...
            *V.R*ZaLo.U(:,j)+0.5*lambda_Lo*(separ_aLo.y(:,j)'*separ_aLo.y(:,j))+...
            YaLo.y(:,j)'*separ_aLo.y(:,j);
        curv_dir_oLo=curv_dir_oLo+Grad_aLo.y(:,j)'*d.y(:,j);
    end
    
    for j=1:Ns
        phiLo=phiLo+tree.prob(non_leaf+j)*ZaLo.X(:,non_leaf+j)'*V.Vf{j}*ZaLo.X(:,non_leaf+j)+...
            Ynew.yt{j}'*separ_aLo.yt{j}+0.5*lambda_Lo*norm(separ_aLo.yt{j})^2;
        curv_dir_oLo=curv_dir_oLo+Grad_aLo.yt{j}'*d.yt{j};
    end
    
    phiLo=-phiLo;
    %curv_dir_oLo=-curv_dir_oLo;
    
    if (phi_c > phi0+ops.c1*alpha_c*ops.curv_dir) || (phi_c >= phiLo)
        %i
        aHo=alpha_c;
    else
        %Grad_c=(Q*Xkk+b)'*d;
        if abs(new_curv_dir)<=-ops.c2*ops.curv_dir
            alpha =alpha_c;
            details_zoom.iter=i;
            return
            %break
        end
        
        if new_curv_dir'*(aHo-aLo)>=0
            aHo=aLo;
        end
        aLo=alpha_c;
    end
    i=i+1;
    if(i>ops.iter_max)
        disp('max iterations zoom')
        alpha=alpha_c;
        details_zoom.term_WF=1;
        details_zoom.iter=ops.iter_max;
        break
    end
end

end
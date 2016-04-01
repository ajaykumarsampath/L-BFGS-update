function [alpha]=zoom_sectioning(obj,Y,d,aLo,aHo,ops)



phi0=ops.phi0;
x0=ops.x0;
i=1;

while(i>0)
    
    alpha_c = 1/2*(aLo+aHo);
    
    % New direction
    Ynew.y=Y.y+alpha_c*d.y;
    Ynew.yt=Y.yt+alpha_c*d.yt;
    
    [Grad_new,Znew,details_new] =obj.grad_dual_envelop(Ynew,x0);
    
    separ_var_new.y=details_new.Hx-details_new.T.y;
    separ_var_new.yt=details_new.Hx_term-details_new.T.yt;
    
    phi_c=0;
    new_curv_dir=0;
    for i=1:non_leaf
        phi_c=phi_c+Znew.X(:,i)'*V.Q*Znew.X(:,i)+Znew.U(:,i)'...
            *V.R*Znew.U(:,i)+0.5*lambda*(separ_var_new.y'*...
            separ_var_new.y)+Ynew.y(:,i)'*separ_var_new.y(:,i);
        new_curv_dir=new_curv_dir+Grad_new.y(:,i)'*d.y(:,i);
    end
    
    for i=1:Ns
        phi_c=phi_c+Znew.X(:,non_leaf+i)'*V.Vf*Znew.X(:,non_leaf+i)+...
            Ynew.yt{i}'*separ_var_new.yt{i}+0.5*lambda*...
            norm(separ_var_new.yt{i})^2;
        new_curv_dir=new_curv_dir+Grad_new.yt{i}'*d.yt{i};
    end
    
     % Lower bound
    YaLo.y=Y.y+aLo*d.y;
    YaLo.yt=Y.yt+aLo*d.yt;
    
    [Grad_aLo,ZaLo,details_aLo] =obj.grad_dual_envelop(YaLo,x0);
    
    separ_aLo.y=details_aLo.Hx-details_aLo.T.y;
    separ_aLo.yt=details_aLo.Hx_term-details_aLo.T.yt;
    
    phiLo=0;
    new_curv_dir=0;
    for i=1:non_leaf
        phiLo=phiLo+ZaLo.X(:,i)'*V.Q*ZaLo.X(:,i)+ZaLo.U(:,i)'...
            *V.R*ZaLo.U(:,i)+0.5*lambda*(separ_aLo.y'*...
            separ_aLo.y)+YaLo.y(:,i)'*separ_aLo.y(:,i);
        new_curv_dir=new_curv_dir+Grad_aLo.y(:,i)'*d.y(:,i);
    end
    
    for i=1:Ns
        phiLo=phiLo+ZaLo.X(:,non_leaf+i)'*V.Vf*ZaLo.X(:,non_leaf+i)+...
            Ynew.yt{i}'*separ_aLo.yt{i}+0.5*lambda*norm(separ_aLo.yt{i})^2;
        new_curv_dir=new_curv_dir+Grad_aLo.yt{i}'*d.yt{i};
    end
    
    if (phi_c > phi0+ops.c1*alpha_c*ops.curv_dir) | (phi_c >=phiLo)
        %i
        aHo=alpha_c;
    else
        %Grad_c=(Q*Xkk+b)'*d;
        if abs(new_curv_dir)<=-ops.c2*ops.curv_dir
            alpha =alpha_c;
            return
            %break
        end
        
        if Grad_c'*(aHo-aLo)>=0
            aHo=aLo;
        end
        aLo=alpha_c;
    end
    i=i+1;
    if(i>ops.iter_max)
        disp('max iterations zoom')
        alpha=alpha_c;
        break
    end
end

end
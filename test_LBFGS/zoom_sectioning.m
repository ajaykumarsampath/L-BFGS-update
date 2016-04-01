function [alpha]=zoom_sectioning(fun_quad,Grad,x,d,aLo,aHo,ops)

% function
Q=fun_quad.Q;
b=fun_quad.b;

phi0=0.5*x'*Q*x+b'*x;

i=1;

%
while(i>0)
    
    alpha_c = 1/2*(aLo+aHo);
    Xkk=x+alpha_c*d;
    
    phi_c=0.5*Xkk'*Q*Xkk+b'*Xkk;
    Grad_c=(Q*Xkk+b)'*d;
    
    xLo=x+aLo*d;
    phiLo=0.5*xLo'*Q*xLo+b'*xLo;
    if (phi_c > phi0+ops.c1*alpha_c*Grad'*d) | (phi_c >=phiLo)
        %i
        aHo=alpha_c;
    else
        %Grad_c=(Q*Xkk+b)'*d;
        if abs(Grad_c')<=-ops.c2*Grad'*d
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
%{

  xLo=x+aLo*d;
  phiLo=0.5*xLo'*Q*xLo+b'*xLo;
    
while(i>0)
    
    alpha_c = 1/2*(aLo+aHo);
    Xkk=x+alpha_c*d;
    
    phi_c=0.5*Xkk'*Q*Xkk+b'*Xkk;
    Grad_c=(Q*Xkk+b)'*d;
    
    if (phi_c > phi0+ops.c1*alpha_c*Grad'*d) | (phi_c >=phiLo)
        %i
        aHo=alpha_c;
    else
        %Grad_c=(Q*Xkk+b)'*d;
        if abs(Grad_c')<=-ops.c2*Grad'*d
            alpha =alpha_c;
            return
            %break
        end
        
        if Grad_c'*(aHo-aLo)>=0
            aHo=aLo;
            aLo=alpha_c;
            phiLo=phi_c;
        end
        %aLo=alpha_c;
    end
    i=i+1;
    if(i>ops.iter_max)
        disp('max iterations zoom')
        alpha=alpha_c;
        break
    end 
end
%}

end
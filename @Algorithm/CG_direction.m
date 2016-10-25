function [ dir_env,beta ] = CG_direction( obj,Grad_env)
%
% This function calculate the direction using conjugate-gradient. Present 
% implementation uses the Fletcher-Reeves CG update. 
%
% Syntax : [ obj,dir_env ] = LBFGS_direction( obj,Grad_env,Grad_envOld,Y,Yold)
%
% Input  :    obj          :   Algorithm object 
%             Grad_env     :   Gradient of the envelop
% 
% Output :    dir_env      :   Direction calculated with LBFGS method
%
%


ConjGrad=obj.algo_details.ops_FBE.ConjGrad;

Ns=length(obj.SysMat_.tree.leaves);


% beta^{FR}=norm(Grad_env)^2/norm(Grad_envOld)
beta=norm([vec(Grad_env.y);cell2mat(Grad_env.yt)])^2/ConjGrad.prev_grad_norm(1,ConjGrad.iterate);

% d_{k+1}=-Grad_env+beta^{FR}d_k
dir_env.y=-Grad_env.y+beta*ConjGrad.prev_dir.y;
for i=1:Ns
    dir_env.yt{i,1}=-Grad_env.yt{i,1}+beta*ConjGrad.prev_dir.yt{i,1};
end 

end


classdef Algorithm
    
    properties
        SysMat_=[];
        Ptree_=[];
        algo_details=[];
    end
    
    
    methods (Access=public)
        function obj = Algorithm(varargin)
            %
            % controller = MPController(SysMat,ops_algorithm);
            %
            if isempty(varargin)
                return; % return empty object
            else
                obj=Algorithm();
            end
            obj.SysMat_=varargin{1};
            if (length(varargin)==2)
                obj.algo_details=varargin{2};
            end
            ops_APG.steps=2000;
            ops_APG.primal_inf=1e-2;
            ops_APG.dual_gap=1e-2;
            
            %ops_APG.alpha=1/calculate_Lipschitz(obj.SysMat_.sys,obj.SysMat_.V,...
            %   obj.SysMat_.tree);
            
            ops_APG.prox_LS='no';
            ops_FBE=ops_APG;
            ops_APG.type='yes';
            %{
            if(~isfield(obj.algo_details,'ops_APG'))
                ops_APG.lambda=obj.calculate_Lipschitz();
            elseif(~isfield(obj.algo_details.ops_APG,'lambda'))
                ops_APG.lambda=obj.calculate_Lipschitz();
            end 
            if(~isfield(obj.algo_details,'ops_FBS'))
                ops_FBS.lambda=obj.calculate_Lipschitz();
            elseif(~isfield(obj.algo_details.ops_FBS,'lambda'))
                ops_FBS.lambda=obj.calculate_Lipschitz();
            end 
            %}
            ops_FBE.memory=5;
            ops_FBE.LS='WOLFE';
             obj.algo_details.ops_FBE.monotonicity='yes';
             
            default_options=struct('algorithm','APG','verbose',0,...
                'ops_APG',ops_APG,'ops_FBE',ops_FBE);
            
            flds = fieldnames(default_options);
            for i=1:numel(flds)
                if (~isfield(obj.algo_details,flds(i)) &&...
                        ~isstruct(default_options.(flds{i})))
                    obj.algo_details.(flds{i})=default_options.(flds{i});  
                else
                    if ~isfield(obj.algo_details,flds(i))
                        obj.algo_details.(flds{i})=default_options.(flds{i});
                    else
                        if(isstruct(default_options.(flds{i})))
                            sub_flds=fieldnames(default_options.(flds{i}));
                            for j=1:numel(sub_flds)
                                if ~isfield(obj.algo_details.(flds{i}),sub_flds(j))
                                    obj.algo_details.(flds{i}).(sub_flds{j})=...
                                        default_options.(flds{i}).(sub_flds{j});
                                end
                            end
                        end
                    end
                end
            end
            
            
            if(strcmp(obj.SysMat_.sys_ops.precondition,'Jacobi'))
               obj.algo_details.ops_FBE.lambda=obj.calculate_Lipschitz();
               obj.algo_details.ops_APG.lambda=obj.calculate_Lipschitz();
            end    
                
            obj.algo_details.ops_FBE.Lbfgs.LBFGS_col = 1;
            obj.algo_details.ops_FBE.Lbfgs.LBFGS_mem = 0;
            obj.algo_details.ops_FBE.Lbfgs.skipCount = 0;
            obj.algo_details.ops_FBE.alphaC=1;
            % Backtracking parameters 
            obj.algo_details.ops_FBE.betaB=0.5;
            obj.algo_details.ops_FBE.alphaB=0.5;
            % Monotonicity option
            obj.algo_details.ops_FBE.monotonicity='yes';
            
            nx=obj.SysMat_.sys.nx;
            nu=obj.SysMat_.sys.nu;
            Ns=length(obj.SysMat_.tree.leaves);
            Nd=length(obj.SysMat_.tree.stage);
            non_leaf=Nd-Ns;
            memory=obj.algo_details.ops_FBE.memory;
            
            dim_dual=2*(non_leaf*(nx+nu)+Ns*nx);
            obj.algo_details.ops_FBE.Lbfgs.S=zeros(dim_dual, memory); % dual variable 
            obj.algo_details.ops_FBE.Lbfgs.Y  = zeros(dim_dual,memory); % dual gradient 
            obj.algo_details.ops_FBE.Lbfgs.YS = zeros(memory, 1); 
            
        end
        
        alpha=calculate_Lipschitz(obj);
        
        obj=Factor_step(obj)
        % This function computes the factor step for the accelerated
        % proximal gradient method. The matrixes calcualted are stored
        % in the structure Ptree
        
        [Z,Q]=Solve_step(obj,Y,xinit)
        % This function computes the dual gradient on-line using the 
        % matrices already computed 
        
        [Y,details_prox]=proximal_gconj(obj,Z,W);
        % This function computes the proximal with respect to the 
        % conjugate g^{\star}
        
        [Z,Y,details_ALG]=Dual_APG(obj,x0)
        % This function is the implementation of the accelerated proximal 
        % gradient method on the dual 
        
        [lambda]=backtacking_prox_method(obj,prev_lambda,ops_backtrack)
        % This function is the implementation of the backtracking algorithm
        % to find the step-size in the proximal gradient mehtod 
        
        [Z,Y,details_ALG]=Solve_step_direction(obj,x0)
        % This function is the implementation of the accelerated proximal 
        % gradient method on the dual 
        
        [ Hd ] = dual_hessian_free( obj,Y,d,Z)
        % This function approximate the dual hessian update function 
        
        [ Grad_env,Z,details] = grad_dual_envelop( obj,Y,x0)
        % This function calcualte the gradient of the envelope of the 
        % dual function 
        
        [ obj,dir_env ] = LBFGS_direction( obj,Grad_env,Grad_envOld,Y,Yold)
        % This function calculates the direction using the
        % limited memory quasi-newton method 
        
        [ Z,Y1,details ] = Dual_FBE(obj,x0)
        % This function implements the L-BFGS method for the Forward-Backward
        % Envelope on the dual function 
        
        [Z,Y1,details]=Dual_FBE_extGrad(obj,x0)
        % This function implements the L-BFGS method for the Forward-Backward
        % Envelope on the dual function. The L-BFGS update is calculated after 
        % the buffer is filled 
        
        [ alpha,term_LS ] = wolf_linesearch( obj,Grad,Z,Y,d,ops)
        % This function calculates the step size for the direction calculated
        % from L-BFGS method. This stepsize should satisfy the 
        % strong wolfe condition. Algorithm 3.2 Nocedal and Wright
        
        [ alpha,term_LS ]= zoom_sectioning(obj,Y,d,aLo,aHo,ops)
        % This function calculates the step size for the direction calculated
        % from L-BFGS method. This stepsize should satisfy the 
        % strong wolfe condition. Algorithm 3.2 Nocedal and Wright
        
        [ alpha,details_LS ] = Goldstein_conditions( obj,Grad,Z,Y,d,ops)
        % This funciton calculates the step size for the direction
        % calculated from L-BFGS method. This step-size should satisfy the 
        % GOldstein conditions
        
        [Z,Y,details]=Dual_GlobalFBE(obj,x0)
        % This function is the implement with an intermediate L-BFGS step that 
        % decrease the cost on the envelope and later apply the proximal
        % gradient method. 
        
        [Y,details_prox]=GobalFBS_proximal_gcong(obj,Z,W)
        % This function is the implementation of the proximal on the
        % conjugate of the dual in the FBE function. 
        
        [ alpha,details_LS ] = LS_backtracking(obj,Grad,Z,Y,d,ops)
        % This function is the implements with an line search method for
        % the decrease of cost on the envelope. 
        
        [Z,Y,details]=Dual_GlobalFBE_version2(obj,x0)
        % This function 
        
        [ alpha,details_LS ] = LS_backtrackingVersion2(obj,Grad,Z,Y,d,ops)
        % This function
        
        [ Lbfgs,dir_env ] = LBFGS_direction_version2( obj,Grad_env,Grad_envOld,Y,Yold)
        % This funciton 
        %
        [Z,Y0,details]=Dual_AccelGlobFBE(obj,x0)
        % This function is the implementation of the L-BFGS step with
        % accelerated step.
        
        [Z,Y0,details]=Dual_AccelGlobFBE_version2(obj,x0)
        % This function is the implementation of the L-BFGS step with
        % accelerated step--Version 2
    end
    
end
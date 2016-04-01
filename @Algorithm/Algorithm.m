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
            ops_APG.primal_inf=5e-3;
            ops_APG.dual_gap=1e-2;
            
            %ops_APG.alpha=1/calculate_Lipschitz(obj.SysMat_.sys,obj.SysMat_.V,...
            %   obj.SysMat_.tree);
            ops_APG.lambda=obj.calculate_Lipschitz();
            ops_FBS=ops_APG;
            ops_FBS.memory=5;
            default_options=struct('algorithm','APG','verbose',0,...
                'ops_APG',ops_APG,'ops_FBS',ops_FBS);
            
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
            
            obj.algo_details.ops_FBS.Lbfgs.LBFGS_col = 1;
            obj.algo_details.ops_FBS.Lbfgs.LBFGS_mem = 0;
            obj.algo_details.ops_FBS.Lbfgs.skipCount = 0;
            obj.algo_details.ops_FBS.alphaC=1;
            
            nx=obj.SysMat_.sys.nx;
            nu=obj.SysMat_.sys.nu;
            Ns=length(obj.SysMat_.tree.leaves);
            Nd=length(obj.SysMat_.tree.stage);
            non_leaf=Nd-Ns;
            memory=obj.algo_details.ops_FBS.memory;
            
            dim_dual=2*(non_leaf*(nx+nu)+Ns*nx);
            obj.algo_details.ops_FBS.Lbfgs.S=zeros(dim_dual, memory); % dual variable 
            obj.algo_details.ops_FBS.Lbfgs.Y  = zeros(dim_dual,memory); % dual gradient 
            obj.algo_details.ops_FBS.Lbfgs.YS = zeros(memory, 1); 
            
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
        
        [ Hd ] = dual_hessian_free( obj,Y,d,Z)
        % This function approximate the dual hessian update function 
        
        [ Grad_env,Z,details] = grad_dual_envelop( obj,Y,x0)
        % This function calcualte the gradient of the envelope of the 
        % dual function 
        
        [ obj,dir_env ] = LBFGS_direction( obj,Grad_env,Grad_envOld,Y,Yold)
        % This function calculates caculates the direction using the
        % limited mehory quasi-newton method; 
        
        [ Z,Y1,details ] = Dual_FBS(obj,x0)
        
        [ alpha ] = wolf_linesearch( obj,Grad,Z,Y,d,ops)
        
        [ alpha ]= zoom_sectioning(obj,Y,d,aLo,aHo,ops)
        
        
    end
    
end
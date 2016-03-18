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
            ops_APG.primal_inf=1e-3;
            ops_APG.dual_gap=1e-2;
            ops_APG.alpha=1/calculate_Lipschitz(obj.SysMat_.sys,obj.SysMat_.V,...
                obj.SysMat_.tree);
            default_options=struct('algorithm','APG','verbose',0,'ops_APG',ops_APG);
            if (isempty(obj.algo_details))
                obj.algo_details=default_options;
            end
        end
           
        obj=Factor_step(obj)
        % This function computes the factor step for the accelerated
        % proximal gradient method. The matrixes calcualted are stored
        % in the structure Ptree
        %
          
        
        [Z,Q]=Solve_step(obj,Y,xinit)
            
        [Z,Y,details_ALG]=Dual_APG_algorithm(obj,x0)
        
    end
    
end
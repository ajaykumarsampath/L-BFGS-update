classdef SysMat
    
    % PUBLIC PROPERTIES
    properties (Access=public)
        
        %System
        sys=[];
        
        % cost function
        V= [];
        
        %Scenario tree
        tree=[];
        
        
        %Options
        sys_ops=[];
        
        %Verbosity (TODO: Move to options)
        verbose=1;
    end % END OF PUBLIC PROPERTIES
    
    methods (Access=public)
        function obj = SysMat(varargin)
            %
            % System is the constructor of this class
            % which is invoked using the following syntax.
            %
            % Syntax:
            % SysMat = SysMat(ops_masses,ops_system);
            %
            % INPUT
            %     ops_masses    : options of the spring-mass system
            %     ops_system    : options of the scenario tree. 
            %
            
            narginchk(0,2);
            if nargin~=0 && nargin~=2,
                error('The constructor of SMPCmatrices expects 0 or 2 input arguments');
            end
            if isempty(varargin)
                return; % return empty object
            else
                obj=SysMat();
            end
            
            ops_masses=varargin{1};
            ops_system=varargin{2};
            
            Nm=ops_masses.Nm;
            ops_sys_masses.nx=2*Nm;
            ops_sys_masses.nu=Nm-1;
            ops_sys_masses.uncertainty=ops_system.uncertainty;
            ops_sys_masses.ops_masses=ops_masses;
            
            ops_tree.N=ops_system.N;
            ops_tree.brch_fact=ones(ops_tree.N,1);
            ops_tree.brch_fact(1:length(ops_system.brch_fact))=ops_system.brch_fact;
            ops_tree.nx=ops_sys_masses.nx;
            Ns=prod(ops_tree.brch_fact);
            %ops.nx=ops_sys_masses.nx;
            
            for i=1:ops_tree.N;
                if(i<=size(ops_system.brch_fact,1))
                    pd=rand(1,ops_tree.brch_fact(i));
                    if(i==1)
                        ops_tree.prob{i,1}=pd/sum(pd);
                        pm=1;
                    else
                        pm=pm*ops_tree.brch_fact(i-1);
                        ops_tree.prob{i,1}=kron(ones(pm,1),pd/sum(pd));
                    end
                else
                    ops_tree.prob{i,1}=ones(Ns,1);
                end
            end
            %tic
            [obj.sys,obj.V,obj.tree]=obj.system_generation(ops_sys_masses,ops_tree);
            %time.tree_formation=toc;
            
        end % END OF CONSTRUCTOR
        
        [sys,V,tree]=system_generation(obj,ops_sys_masses,ops_tree);
        
        obj=Precondtion_system(obj);
        
    end
    
    
    methods(Static, Access=private)
        
        
    end % END OF STATIC PRIVATE METHODS
    
end % END OF CLASS


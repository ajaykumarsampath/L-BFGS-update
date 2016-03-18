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
        
        function [sys,V,tree]=system_generation(obj,ops_system,ops_tree)
            %
            % This function generated a random from the distribution with
            % braching_factor specified
            % we assume that the tree have a fixed probability that is given.
            % generate the system
            %
            % INPUT : ops_system  N : Number  of mases in the spring-mass system
            %                    ops_masse: spring-const of the spring, damper, sampling time,etc
            %                    uncer_system: If this selected then the system have
            %                    multiplicative disturbace else no
            %
            %         ops_tree   Np: prediction horizon
            %                    branching factor: branching factor of the tree.
            %
            % OUTPUT : sys   : system matrices at each node given by [A(i) B(i)]
            %                  constraints at each node [F(i) G(i)]
            %
            %
            
            
            N=ops_system.nu+1;
            default_options = struct('M', 1*ones(N,1), 'b', 0.1*ones(N+1,1),...
                'k',1*ones(N+1,1),'xmin',-5*ones(2*N,1), 'xmax', 5*ones(2*N,1), 'umin', ...
                -5*ones(N-1,1),'umax',5*ones(N-1,1), 'Ts', 0.1);
            ops = default_options;
            if isfield(ops_system,'ops_masses'), % User-provided options
                ops = ops_system.ops_masses;
                flds = fieldnames(default_options);
                for i=1:numel(flds),
                    if ~isfield(ops_system.ops_masses, flds(i))
                        ops.(flds{i})=default_options.(flds{i});
                    end
                end
            end
            
            M=ops.M;        % M(i)=mass of body #i
            b=ops.b;        % b(i)=viscous friction of body #i
            k=ops.k;        % k(i)=spring of body #i
            
            
            % Define full A,B model
            nx=ops_system.nx;
            nu=ops_system.nu;
            S.nx=nx;
            S.nu=nu;
            
            
            Ag=zeros(nx,nx);
            Bg=zeros(nx,nu);
            for i=1:N,
                h=2*i-1;
                Ag(h,h+1)=1;  % velocity
                Ag(h+1,h+1)=-(b(i)+b(i+1))/M(i);  % friction
                Ag(h+1,h)=-(k(i)+k(i+1))/M(i);    % self-springs
                if i>1,
                    Ag(h+1,h-2)=k(i)/M(i);
                    Ag(h+1,h-1)=b(i)/M(i);
                end
                if i<N,
                    Ag(h+1,h+2)=k(i+1)/M(i);
                    Ag(h+1,h+1)=b(i+1)/M(i);
                end
                if i>1
                    Bg(h+1,i-1)=-1/M(i);
                end
                if i<N
                    Bg(h+1,i)=1/M(i);
                end
            end
            
            sysgc=ss(Ag,Bg,eye(nx),zeros(nx,nu));
            sysgd=c2d(sysgc, ops.Ts);
            tol=1e-6;

            S.A=(1-(abs(sysgd.a)<tol)).*sysgd.a;
            S.B=(1-(abs(sysgd.b)<tol)).*sysgd.b;
            S.F=[eye(S.nx);-eye(S.nx);zeros(2*S.nu,S.nx)];
            S.G=[zeros(2*S.nx,S.nu);eye(S.nu);-eye(S.nu)];
            S.g=[ops.xmax;-ops.xmin;ops.umax;-ops.umin];
            
            tree = struct('stage',0,'value',zeros(1,ops_system.nx),'prob',1,...
                'ancestor',0,'children',cell(1,1),'leaves',1);
            K=1;
            %nodes=1;
            for i=1:ops_tree.N
                %%
                K=K*ops_tree.brch_fact(i);
                q=tree.leaves(end)+1:tree.leaves(end)+K;
                tree.stage(q,1)=i*ones(K,1);
                for j=1:size(tree.leaves,2)
                    
                    l=(j-1)*ops_tree.brch_fact(i)+1:j*ops_tree.brch_fact(i);
                    tree.prob(q(l),1)=tree.prob(tree.leaves(j))*ops_tree.prob{i,1}(j,:);
                    tree.value(q(l),:)=0.1*rand(ops_tree.brch_fact(i),ops_tree.nx);
                    
                    if(strcmp(ops_system.uncertainty,'parametric'))
                        if(i==1)
                            sys.A{i,1}=S.A;
                            sys.B{i,1}=S.B;
                        end
                        for ii=1:size(q,2)
                            bb=b+0.1*rand(1)*b;
                            for kk=1:N
                                h=2*kk-1;
                                Ag(h+1,h+1)=-(bb(kk)+bb(kk+1))/M(kk);
                                if kk>1,
                                    Ag(h+1,h-1)=bb(kk)/M(kk);
                                end
                                if kk<N,
                                    Ag(h+1,h+1)=bb(kk+1)/M(kk);
                                end
                            end
                            %if(i<ops_tree.N)
                            sysgc=ss(Ag,Bg,eye(nx),zeros(nx,nu));
                            sysgd=c2d(sysgc, ops.Ts);
                            S.A=(1-(abs(sysgd.a)<tol)).*sysgd.a;
                            S.B=(1-(abs(sysgd.b)<tol)).*sysgd.b;
                            
                            sys.A{q(ii),1}=S.A;
                            sys.B{q(ii),1}=S.B;
                        end
                        %end
                    else
                        if(i==1)
                            sys.A{i,1}=S.A;
                            sys.B{i,1}=S.B;
                        end
                        for ii=1:size(q,2)
                            sys.A{q(ii),1}=S.A;
                            sys.B{q(ii),1}=S.B;
                        end
                    end
                    tree.ancestor(q(l),1)=tree.leaves(j)*ones(ops_tree.brch_fact(i),1);
                    tree.children{tree.leaves(j),1}=q((j-1)*ops_tree.brch_fact(i)+1:...
                        j*ops_tree.brch_fact(i));
                end
                tree.leaves=q;
            end
            
            sys.nx=S.nx;
            sys.nu=S.nu;
            % stage constraints 
            for i=1:size(tree.children)
                sys.F{i}=S.F;
                sys.G{i}=S.G;
                sys.g{i}=S.g;
            end
            
            % terminal constraints 
            Ns=length(tree.leaves);
            sys.Ft=cell(Ns,1);
            sys.gt=cell(Ns,1);
            
            for i=1:Ns
                sys.Ft{i}=[eye(sys.nx);-eye(sys.nx)];
                sys.gt{i}=(3+0.1*rand(1))*ones(2*sys.nx,1);
                P=Polyhedron('A',sys.Ft{i},'b',sys.gt{i});
                if(isempty(P))
                    error('Polyhedron is empty');
                end
            end
            
            % Stage cost function
            V.Q=eye(sys.nx);
            V.R=eye(sys.nu);
            
            % Terminal cost function
            V.Vf=cell(Ns,1);
            %r=rand(Ns,1);
            r=ones(Ns,1);
            for i=1:Ns
                V.Vf{i}=dare(sys.A{1},sys.B{1},r(i)*V.Q,r(i)*V.R);
            end
            
            sys.Np=ops_tree.N;
        end
        
        sys=Precondtion_system(obj);
        
    end
    
    
    methods(Static, Access=private)
        
        
    end % END OF STATIC PRIVATE METHODS
    
end % END OF CLASS


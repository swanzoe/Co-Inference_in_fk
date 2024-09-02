% INPUT: 
%  J :                      number of frams 
%  F_fun :                  forward operators
%  y :                      vectorized spectra  
%  variance :               list of noise variances 
%  R_fun :                  regularization operator (function)
%  r, beta, vartheta :      regualrization hyper-hyper-parameters 
%
% OUTPUT: 
%  x :                      list of vectorized MAP estimates 
%  theta :                  vector of MAP estimate for theta 
%  history :                structure containing residual norms and the tolerances at each iteration
%  conver_dairy :           convergence diary
function [x, theta, history, conver_dairy] = CoInference( J, F_fun, y, noise_var, R_fun, r, beta, vartheta, QUIET )

    t_start = tic; % begin time count 

    %% Global constants and defaults  
    MAX_ITER = 1000; 
    TOL_x = 1e-6;
    
    %% Data preprocessing & initial values 
    for j=1:J 
        M{j} = length(y{j});                              % number of measurements 
        y_j = y{j};
        ForwardF{j} = F_fun{j}(y_j,'transp');             % forward operator applied to the data 
        N = length( ForwardF{j} );                        % number of parameters 
        x{j} = zeros(N,1); x_OLD{j} = zeros(N,1);         % initialize parameter vectors 
        FtF{j} = @(x) F_fun{j}( F_fun{j}(x,'notransp'), 'transp' ); % product corresponding to the forward operator 
    end
    K = length( R_fun(x_OLD{1},'notransp') );             % number of outputs of the regularization operator 
    theta = ones(K,1);                                    % initial value for theta
   
    %% Outputting the learning progress
    if ~QUIET
        % The first rel is inf
        fprintf('%3s\t%10s\t%10s\n', 'iter', 'max. rel. change in x', 'tol x');
    end
    
    %% Iterations 
    for counter = 1:MAX_ITER

        % 1) Update: X matrix 
        parfor j=1:J
            A_fun{j} = @(x) (1/noise_var{j})*FtF{j}(x) + R_fun( (theta.^(-1)).*R_fun(x,'notransp'), 'transp'); % forward operator 
            [x{j},flag] = pcg( A_fun{j}, (1/noise_var{j})*ForwardF{j},[],[],[],[],x_OLD{j});
        end
        % 2) Update: parameters 
        aux = 0;
        for j=1:J 
            aux = aux + real( R_fun(x{j},'notransp') ).^2; 
        end
        eta = r*beta - (J/2 + 1); %eta
        if r==1 
            theta = 0.5*vartheta*( eta + sqrt( eta^2 + 2*aux/vartheta ) ); 
        elseif r==-1 
            theta = ( aux/2 + vartheta )/( -eta ); 
        else 
            error('Only r=1 and r=-1 are implemented yet!'); 
        end

        % store values in history vlog 
        history.change_x(counter) = 0; 
        change_x = zeros(J,1); 
        % parallel process
        parfor j=1:J 
            change_x(j) = norm( x{j} - x_OLD{j} )/norm( x_OLD{j} ); % relative change in x{j}
            x_OLD{j} = x{j}; % store value of mu 
        end
        max_change = max(change_x); % maximum change 
        history.change_x(counter) = max_change; % sum of relative changes

        % display convergence diary
        if ~QUIET
            fprintf('%3d\t%0.2e\t%0.2e\n', counter, history.change_x(counter), TOL_x);
        end
        conver_dairy(counter) = history.change_x(counter);
        % check for convergence 
        if ( history.change_x(counter) < TOL_x ) 
             break;
        end
        
    end
    
    if ~QUIET
        toc(t_start);
    end
    
end
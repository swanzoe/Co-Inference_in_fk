% Function for operator generation
% total variation: TV operator
%
% INPUT: 
%  N :      number of pixels in each direction 
%  x :      input vector 
%  type :   type of the regularization operator (TV or PA)
%  order :	order of the TV operator 
%
% OUTPUT: 
%  y :    	output vector


function y = ReOperator( N, kelm, x, type, order, flag )

    % Compute the univariate regularization operator matrix 
    if strcmp(type,'TV')   
        R = TV_operator( N, kelm, order ); % TV operator
    else 
        error('Desired regualrization operator not yet implemented!'); 
    end

    R_tr = R'; % Transpose of the univariate operator  
    K = size(R,1); % its output size

    if strcmp(flag,'notransp') % compute R*x 
        x1 = reshape(x,N,kelm);
        y_hor = R*x1'; % vertical/column differences  
        y_ver = x1*R_tr; % vertical differences  
        y = [y_hor(:);y_ver(:)]; % vectorize and stack together 
        %y = sqrt(y_hor(:).^2 + y_ver(:).^2);
        x = x1;
    elseif strcmp(flag,'transp') % compute R'*x 
        z1 = reshape( x(1:K*N), K, N ); 
        z2 = reshape( x(K*N+1:end), N, K );
        y = (R_tr*z1)' + z2*R; 
        y = y(:); % vectrize
        %y = real(y);

    end

end
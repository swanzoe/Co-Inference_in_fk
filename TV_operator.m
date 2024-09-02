% Description: 
%  Function to compute the one-dimensional TV operator 
% 
% INPUT: 
%  n :    	number of equidistant grid points 
%  order :	order of the TV operator 
%
% OUTPUT: 
%  R :    	matrix representation of the TV operator 

function R = TV_operator( n, kelm, order )

    e = ones(n,1);
    % 提取非零对角线并创建稀疏带状对角矩阵
    if order == 1
        D = spdiags([e -e], 0:1, n,kelm);
 	elseif order == 2
       	D = spdiags([-e 2*e -e], 0:2, n,kelm);
 	elseif order == 3
      	D = spdiags([e -3*e 3*e -e], 0:3, n,kelm); 
    else 
        error('Desried order not yet implemented!')
    end
    
    R = D(1:n-order,:);
        
end
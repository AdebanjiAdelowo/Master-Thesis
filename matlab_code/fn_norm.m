% Compute the L^p norm of the spatial slice of u
% Call like fn_norm( u, p, M ).
% u has time in the last argument. M is a matrix multiplier.
function f = fn_norm( varargin )
    u = varargin{1};
    if nargin >= 2;
	p = varargin{2};
    else
	p = 2;
    end;

    % Time in the last dimension. Bring it to the first.
    u = shiftdim( u, ndims(u) - 1 );

    tsize = size( u, 1 );
    f = zeros( tsize, 1 );

    for t = 1:tsize
	if nargin >= 3;
	    M = varargin{3};
	    ut = reshape( u(t, :), size(M) ) .* M;
	else
	    ut = u(t, :);
	end

	f(t) = norm( ut(:), p );
    end
end

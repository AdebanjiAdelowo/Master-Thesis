% Return (\sin(2\pi mx) \sin(\pi my)) \Chi_B
% where B is a square of side length a.
function f = idata_diag( a )
    global N del_x del_y lap_inv;
    global dx xx yy;

    f1 = sin( 2*pi *xx / a) .* sin( 2*pi*yy / a ) .* (xx < a/2 & yy < a / 2);
    f2 = sin( 2*pi *xx / a) .* sin( 2*pi*yy / a ) .* ...
	    (a/2 < xx & xx < a & a/2 < yy & yy < a );

    % Break symmetry by shifting;
    n_rows = floor( N * a/8 );
    f = circshift( f1, [n_rows, 0] ) - circshift( f2, [-n_rows, 0] );
    f = f / ( norm( f(:), 2 ) * dx);

    % Shift data to the center.
    x_shift = floor( N * (1 - a) / 2 );
    y_shift = x_shift;
    f = circshift( f, [ y_shift, x_shift] );

    %f_hat = fft2( f );
    % Normalize f to have L^2 norm 1.
    % For the 2D DFT, norm( f_hat ) = N^2 norm(f).
    %f_hat = f_hat / norm( f_hat(:), 2 ) * N^2;
end

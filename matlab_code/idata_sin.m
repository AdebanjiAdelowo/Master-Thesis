% Return (\sin(2\pi mx) \sin(2\pimy)) \Chi_B
% where B is a square of side length a.
function f = idata_sin( a )
    global N del_x del_y lap_inv;
    global dx xx yy;

    f = sin( 2*pi *xx / a) .* sin( 2*pi*yy / a ) .* (xx < a & yy < a);
    f = f / ( norm( f(:), 2 ) * dx);

    % Shift data to the center.
    space_shift = floor( N * (1 - a) / 2 );
    f = circshift( f, [ space_shift, space_shift, 0] );

    %f_hat = fft2( f );
    % Normalize f to have L^2 norm 1.
    % For the 2D DFT, norm( f_hat ) = N^2 norm(f).
    %f_hat = f_hat / norm( f_hat(:), 2 ) * N^2;
end

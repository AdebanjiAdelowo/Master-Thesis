function [value, isterminal, direction] = res_check(t, y_hat)
    global N dx sqrt_dx sqrt_sqrt_dx tol l4norm_init l8norm_init;

    direction = 1;
    isterminal = 1;

    y = ifft2( reshape( y_hat, [N, N] ), 'symmetric' );

    % Use the L2, L4 and L8 norms as a resolution check. All are conserved, so
    % once they start changing we're in trouble.

    l2 = norm(y(:), 2 ) * dx;
    l4 = norm(y(:), 4 ) * sqrt_dx / l4norm_init;
    l8 = norm(y(:), 8 ) * sqrt_sqrt_dx / l8norm_init;

    value = max( [ abs( l2 - 1 ), abs( l4  - 1 ), abs( l8 - 1 ) ] ) - tol;

    disp( sprintf( 't=%.3f, l2=%f, l4=%f, l8=%f', t, l2, l4, l8) );
end

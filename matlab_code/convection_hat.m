function d_theta_hat = convection_hat( t, theta_hat_vec )
    global F N tol del_x del_y lap_inv;

    % theta_hat = Fourier transform of Theta given in ODE call.
    % t = time
    % return Fourier Transform of -u \cdot \grad theta 
    % where v = -\lap \inv P ( \theta \cdot \grad \inv \theta )
    % and u = v / \norm{\grad v}_{L^2}

    theta_hat = reshape( theta_hat_vec, N, N );

    % g = \theta \grad\inv \theta
    % To convolve accurately we need to "circularly" convolve, which matlab
    % doesn't have in 2D. Instead we ifft, multipliy and fft back.
    linv_theta_hat = lap_inv .* theta_hat;
    theta = ifft2( theta_hat, 'symmetric' );
    g_hat(:, :, 1) = fft2( theta .* ifft2( del_x .* linv_theta_hat, 'symmetric' ) );
    g_hat(:, :, 2) = fft2( theta .* ifft2( del_y .* linv_theta_hat, 'symmetric' ) );

    % div_g_hat = (\divergence g)^\hat
    div_g_hat = del_x .* g_hat(:, :, 1) + del_y .* g_hat( :, :, 2 );

    % v = -\lap\inv P g
    v_hat(:, :, 1) = -lap_inv .* ( g_hat(:,:,1) - del_x .* lap_inv .* div_g_hat );
    v_hat(:, :, 2) = -lap_inv .* ( g_hat(:,:,2) - del_y .* lap_inv .* div_g_hat );

    v_hat_norm1 = del_x.*v_hat(:,:,1);
    v_hat_norm2 = del_x.*v_hat(:,:,2);
    v_hat_norm3 = del_y.*v_hat(:,:,1);
    v_hat_norm4 = del_y.*v_hat(:,:,2);

    norm_v_hat = sqrt( norm( v_hat_norm1(:), 2 )^2 + ...
	norm( v_hat_norm2(:), 2 )^2 + norm( v_hat_norm3(:), 2 )^2 + ...
	norm( v_hat_norm4(:), 2 )^2 );
    norm_v = norm_v_hat / N^2; % L2 conversion factor for DFT.

    % Compare norm_v relative to norm_g
    linv_g_hat( :, :, 1 ) = lap_inv .* g_hat(:, :, 1);
    linv_g_hat( :, :, 2 ) = lap_inv .* g_hat(:, :, 2);
    norm_linv_g = norm( linv_g_hat(:), 2 ) / N^2;

    if norm_v < norm_linv_g *tol;
	disp( sprintf( ...
	    'Possible saddle point: t=%.3f, norm_v=%.9f, norm_g=%.9f', ...
	    t, norm_v, norm_linv_g ) );
    end

    u(:, :, 1) = F*ifft2( v_hat(:, :, 1), 'symmetric' ) / norm_v;
    u(:, :, 2) = F*ifft2( v_hat(:, :, 2), 'symmetric' ) / norm_v;

    d_theta_hat = -fft2( ...
	u(:,:,1) .* ifft2( del_x.*theta_hat, 'symmetric' ) + ...
	u(:,:,2) .* ifft2( del_y.*theta_hat, 'symmetric' ) );

    % Finally we need our output to be used in ode23 or ode45 functions, so we
    % need reshape it.
    d_theta_hat = d_theta_hat(:);
end

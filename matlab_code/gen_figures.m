% Solve \delt \theta + (u \cdot \grad) \theta = 0
% Where u = v / \norm{\grad v}_{L^2}
% and v = -\lap \inv P ( \theta \cdot \grad \inv \theta ).
% This is the optimal mixing velocity by Lin-Thiffeault-Doering


tic
clearvars -EXCEPT F N t_int a_range idata tol

% 
% Parameters (Any parameter not set before running is set here.)
%

% # spectral modes in each variable to retain. Powers of 2 are faster.
if ~ exist('N'); global N; N = 64; end

% Enstrophy constraint: enforce \norm{\grad u(t)} = F
if ~ exist('F'); global F; F = 1; end

% Time interval. (We will certainly run out of resolution before this time).
if ~ exist('t_int'); t_int = 0:.05:10; end

% Range of scaling parameter "a". (a=1 is forbidden, if theta0 = sin x sin y)
% Values in a_range should be multiples of 1/N;
if ~ exist('a_range');	a_range = .5:1/16:15/16; end

% Initial data function
if ~ exist('idata'); idata = @idata_sin; end

% Error tolerance. If conserved quantities increase by more than this, then stop.
if ~ exist('tol'); global tol; tol = 1e-3; end


%
% End parameters
%


% Fourier multiplier matrices.
% The x derivative corresponds to multipliying the k'th COLUMNS by (2 pi i k)
% when k < N/2, and by (2 pi i k - N) when k > N/2
% The y derivative does the same on the rows.

global del_x del_y lap_inv

k  = 0:N-1;
k_matrix = 2i*pi* diag(k - N * (k > N/2) );
del_x = ones(N) * k_matrix;
del_y = k_matrix * ones(N);

lap_inv = 1 ./ (del_x.^2 + del_y.^2);
lap_inv( 1, 1 ) = 0;

lambda_inv = sqrt( -lap_inv );

% For N even we have to multipliy the N/2'th fourier modes by 0 for the first
% derivative. For the second derivative we don't have to do this. (Google fft
% differentiation for why).
if mod( N, 2 ) == 0
    del_x = del_x * diag( k ~= N/2 );
    del_y = diag( k ~= N/2 ) * del_y;
end

% Store norms of the computed solution in here
ncells = size( a_range, 2 );
t = cell( ncells, 1 );
norm_hminus1 = cell( ncells, 1 );
norm_l2	     = cell( ncells, 1 );
norm_l4	     = cell( ncells, 1 );
norm_l8	     = cell( ncells, 1 );

% Passed to res_check for resolution checking.
global dx sqrt_dx sqrt_sqrt_dx xx yy l4norm_init l8norm_init;

% For plotting
dx = 1/N;
sqrt_dx = sqrt(dx);
sqrt_sqrt_dx = dx^.25;

[xx, yy] = meshgrid( 0:dx:1-dx );

cc=lines( ncells );

i = 0;
for a = a_range
    i = i+1;

    theta0 = idata(a);
    theta0_hat = fft2( theta0 );

    l4norm_init = norm( theta0(:), 4 ) * sqrt_dx;
    l8norm_init = norm( theta0(:), 8 ) * sqrt_sqrt_dx;

    % Not sure what solver to try; Can't really do ode15s, since I've no idea
    % what the Jacobian is. ode45 works well for short times.
    options = odeset( 'Events', @res_check );
    [t{i}, theta_hat] = ode45( @convection_hat, t_int, theta0_hat, options );

    % Reshape for convenience.
    tsize = size( t{i}, 1);
    theta_hat = reshape( shiftdim( theta_hat, 1 ), [N, N, tsize] );

    % Compute theta, so we can plot it and keep track of L^p norms
    theta = ifft2( theta_hat, 'symmetric' );

    % Resolution check norms
    % DFT norm conversion: norm( f_hat ) = N^2 norm( f ).
    norm_l2{i} = fn_norm( theta_hat ) / N^2;
    norm_l4{i} = fn_norm( theta, 4 );
    norm_l8{i} = fn_norm( theta, 8 );

    norm_l4{i} = norm_l4{i} / norm_l4{i}(1);
    norm_l8{i} = norm_l8{i} / norm_l8{i}(1);


    % Mix norm. (We normalize by the initial mix norm)
    norm_hminus1{i} = fn_norm( theta_hat, 2, lambda_inv );
    norm_hminus1{i} = norm_hminus1{i} / norm_hminus1{i}(1, 1);


    % Plot figures.
    figure(1);
    if i == 1; hold off; else; hold on; end;
    plot( t{i}, log( norm_hminus1{i} ), 'color', cc(i,:) );
    %title( 'Log mix-norm vs t' );

    figure(2);
    hold off;
    plot( t{i}, norm_l2{i}, 'r' );
    hold on;
    plot( t{i}, norm_l4{i}, 'g' );
    plot( t{i}, norm_l8{i}, 'b' );
    title( 'Lp norms (resolution check)' );

    % Solution plots to see if we're actually mixing
    nfigs = 6;
    for j=0:nfigs-1;
	figure( 4 + j );
	t_ind = max( floor( j * tsize / (nfigs-1) ), 1 );
	%contourf( xx, yy, theta(:, :, t_ind) );
	pcolor( xx, yy, theta(:, :, t_ind) );
	shading interp;
	title( sprintf( 't = %.3f', t{i}( t_ind ) ) );

	% Save the solution, in case we want it later.
	theta_sample{i}(:, :, j+1) = theta( :, :, t_ind );
    end % for j
    drawnow
    toc
end %for m

% Fit all slopes of the mix norm log plots to a line, and plot vs m.
figure(3);
clf;
for i = 1:ncells
    p = polyfit(t{i}, log( norm_hminus1{i} ), 1);
    slope(i) = p(1);
end
plot( a_range, -1 ./ slope, '*' );

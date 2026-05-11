nfigs = size( theta_sample{1}, 3 );
cc = lines(ncells);

figure(1);
clf;
hold on;
for j=1:ncells;
    plot( t{j}, log( norm_hminus1{j} ), 'color', cc(j,:) );
end

figure(2);
clf;
hold on;
cc = lines(ncells);
for j=1:ncells;
    plot( t{j}, norm_hminus1{j}, 'color', cc(j,:) );
end

figure(3);
clf
for j = 1:ncells
    ntimes = size( t{j}, 1);
    t_trunc = floor( ntimes / 3 );
    p = polyfit(t{j}(t_trunc+1:end), ...
	    log( norm_hminus1{j}(t_trunc+1:end) ), 1);
    slope(j) = p(1);
end
plot( a_range, -1./slope, '*' );

% Compute how the decay rate changes with a.
if( size( a_range, 2 ) > 3 )
    p = polyfit( log( a_range ), log( -slope ), 1 );
    disp( sprintf( 'Computed decay rate: a^%f. (Predicted a^-1)', p(1) ) );

    p = polyfit( a_range, -1 ./ slope, 1 );
    hold on;
    plot( a_range, a_range * p(1) + p(2), '--r' );
end

if ~exist('i', 'var'); i = 1; end
tsize = size( t{i} );
for j = 0:nfigs-1;
    figure( 4 + j );

    pcolor( xx, yy, theta_sample{i}(:, :, j+1) );
    shading interp;

    axis off;
    set( gca, 'Position', [0, 0, 1, 1] );

    t_ind = max( [floor( j * tsize / (nfigs-1) ), 1 ] );
    %title( sprintf( 't = %.3f', t{i}( t_ind ) ) );
end

% Save important stuff into a data file.
filename=input( 'Filename to save data: ', 's' );

save( filename, 'N', 'F', 't_int', 'a_range', 't', 'xx', 'yy', 'dx', ...
    'idata', 'norm_hminus1', 'theta_sample', 'ncells' );

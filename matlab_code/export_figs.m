% Export figures to EPS files

for i = 1:9;
    figure(i);
    set(gcf, 'PaperSize', [4, 4], 'PaperPosition',[0,0,4,4]);
end

figure(1);
print( 'figures/log_mix_norm.eps', '-depsc' );

figure(2);
print( 'figures/mix_norm.eps', '-depsc' );

figure(3);
print( 'figures/decay_rate.eps', '-depsc' );

for i=4:9;
    figure(i);
    print( sprintf( 'figures/sol%d', i-3 ), '-depsc' );
end

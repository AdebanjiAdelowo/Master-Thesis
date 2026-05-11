% Export figures to JPEG pics

for i = 1:9;
    figure(i);
    set(gcf, 'PaperSize', [4, 4], 'PaperPosition',[0,0,4,4]);
end

figure(1);
print( 'pics/log_mix_norm.jpg', '-djpeg90' );

figure(2);
print( 'pics/mix_norm.jpg', '-djpeg90' );

figure(3);
print( 'pics/decay_rate.jpg', '-djpeg90' );

for i=4:9;
    figure(i);
    print( sprintf( 'pics/sol%d.jpg', i-3 ), '-djpeg90' );
end

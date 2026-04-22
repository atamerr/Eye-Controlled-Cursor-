x=0:0.1:10;

subplot(3,1,1);
plot(x,cos(x),'r--','LineWidth',2);
title('cos');
xlabel('X');
text(pi, 0, '\pi');
ylabel('cos(x)');
text(pi, 0, '\pi'); 
grid on;


subplot(3,1,2);
plot(x,sin(x),LineWidth=4,Color='blue')
title('sin');
xlabel('x');
ylabel('sin(x)');
grid on;

subplot(3,1,3);
plot(x,tan(x),LineWidth=2,Color='black');
title('tan');
xlabel('x');
ylabel('tan(x)');
grid on;
axis([0 2*pi -5 5]); 
clc, clear;
figure(1)
x = 0:2 * pi/30 : 2 * pi;
plot(x, sin(x), 'r', 'LineStyle', '-.');
grid on;
xlabel('x');
ylabel('y');
title('y = sin(x)');

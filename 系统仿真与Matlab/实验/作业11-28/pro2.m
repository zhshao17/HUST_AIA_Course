clc,clear;
figure(1)
x = -2 * pi: pi /100: 2 * pi ;
y1 = sin(x);
y2 = x;
fplot('sin(x)', [-2 * pi, 2 * pi], 'b', 'LineStyle', ':');
hold on
plot(x, y2, 'r',  'LineStyle', '-.');
hold on
fplot('tan(x)', [-2 * pi, 2 * pi], 'g',  'LineStyle', '-');
legend('y = sin(x)', 'y = x', 'y = tan(x)');
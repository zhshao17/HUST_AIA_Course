function [] = draw(~)
f = input('input your function', "s")
x = 0:2*pi/100:2 * pi;
switch f
    case 'sin'
       y = sin(x);
       plot(x, y),title('y=sin(x)');
    case 'cos'
        y = cos(x);
        plot(x, y),title('y=cos(x)');
    case 'tan'
        y = tan(x);
        plot(x, y),title('y=tan(x)');
    otherwise
        print('your input is unkwon!')
end
end

    
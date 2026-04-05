clc;
clear;
close all;

%% Вхідні дані
% Кожен стовпець — окремий образ [x1; x2; x3]
P = [-3  3 -3  3 -3  3 -3  3;
     -3 -3  3  3 -3 -3  3  3;
     -3 -3 -3 -3  3  3  3  3];

% Цільові класи
T = [0 1 1 1 0 0 0 1];

%% Створення одношарового персептрона
net = perceptron;

% Налаштування параметрів навчання
net.trainParam.epochs = 100;

%% Навчання мережі
net = train(net, P, T);

%% Перевірка результату
Y = net(P);

disp('Очікувані класи T:');
disp(T);

disp('Вихід мережі Y:');
disp(Y);

disp('Ваги мережі w:');
disp(net.IW{1,1});

disp('Зміщення b:');
disp(net.b{1});
figure;
hold on;
grid on;
box on;
view(3);

%% Точки класу 0
idx0 = find(T == 0);
plot3(P(1, idx0), P(2, idx0), P(3, idx0), 'bo', ...
    'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'b');

%% Точки класу 1
idx1 = find(T == 1);
plot3(P(1, idx1), P(2, idx1), P(3, idx1), 'rs', ...
    'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'r');

%% Отримання ваг і зміщення
w = net.IW{1,1};   % [w1 w2 w3]
b = net.b{1};

%% Побудова площини розділення
% Рівняння площини:
% w1*x + w2*y + w3*z + b = 0
[x, y] = meshgrid(-4:0.5:4, -4:0.5:4);

if abs(w(3)) > 1e-6
    z = -(w(1)*x + w(2)*y + b) / w(3);
    surf(x, y, z, 'FaceAlpha', 0.4, 'EdgeColor', 'none');
else
    disp('Неможливо побудувати площину у вигляді z=f(x,y), бо w(3)=0');
end

%% Оформлення
xlabel('x_1');
ylabel('x_2');
zlabel('x_3');
title('Класифікація трьохелементних образів одношаровим персептроном');
legend('Клас 0', 'Клас 1', 'Площина розділення', 'Location', 'best');
axis equal;
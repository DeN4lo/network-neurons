clc;
clear;
close all;

% Вхідні вектори (кожен стовпець - окремий образ)
P = [1.0  1.0  2.0  2.0;
     1.0 -1.0  0.5 -2.0];

% Цільові класи
% 1 - перший клас
% 0 - другий клас
T = [1 1 0 0];

% Створення одношарового персептрона
net = perceptron;

% Налаштування мережі під вхідні та цільові дані
net = configure(net, P, T);

% Параметри навчання
net.trainParam.epochs = 100;

% Навчання мережі
net = train(net, P, T);

% Перевірка результату
Y = net(P);

disp('Результат класифікації:');
disp(Y);

disp('Ваги мережі:');
disp(net.IW{1,1});

disp('Зсув (bias):');
disp(net.b{1});

% ---------------------------------------------------
% Побудова графіка
% ---------------------------------------------------
figure;
hold on;
grid on;

% Клас 1 (зелений)
idx1 = find(T == 1);
plot(P(1, idx1), P(2, idx1), 'go', ...
    'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'g');

% Клас 0 (фіолетовий)
idx0 = find(T == 0);
plot(P(1, idx0), P(2, idx0), 'md', ...
    'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'm');

% Ваги і зміщення
w = net.IW{1,1};
b = net.b{1};

% Лінія розділення (помаранчева)
x = linspace(0, 3, 100);

if abs(w(2)) > 1e-6
    y = -(w(1)*x + b) / w(2);
    plot(x, y, 'Color', [1 0.5 0], 'LineWidth', 3); % помаранчевий
else
    x_line = -b / w(1);
    xline(x_line, 'Color', [1 0.5 0], 'LineWidth', 3);
end

xlabel('x_1');
ylabel('x_2');
title('Простір входів та лінія розділення');
legend('Клас 1', 'Клас 0', 'Межа розділення', 'Location', 'best');

hold off;
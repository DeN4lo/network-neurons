clc;
clear;
close all;

%% Вхідні дані
% Кожен стовпець — окремий образ [x1; x2]
P = [ 0.1   0.7   0.8   0.8   1.0   0.3   0.0  -0.3  -0.5  -1.5;
      1.2   1.8   1.6   0.6   0.8   0.5   0.2   0.8  -1.5  -1.3 ];

% Цільові класи у двобітному вигляді
% [1;0], [0;0], [1;1], [0;1]
T = [ 1  1  1  0  0  1  1  1  0  0;
      0  0  0  0  0  1  1  1  1  1 ];

%% Створення нейронної мережі
% 2 входи -> прихований шар (8 нейронів) -> 2 виходи
net = patternnet(8);

% Функції активації
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'logsig';

% Параметри навчання
net.trainFcn = 'trainscg';
net.performFcn = 'crossentropy';

% Співвідношення вибірок
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio   = 0.1;
net.divideParam.testRatio  = 0.1;

% Навчання мережі
[net, tr] = train(net, P, T);

%% Тестування на навчальних даних
Y = net(P);
Ybin = round(Y);

disp('Виходи мережі після округлення:');
disp(Ybin);

%% Перетворення двобітного коду у номер класу
% [0;0] -> 1
% [0;1] -> 2
% [1;0] -> 3
% [1;1] -> 4
targetClass = vec2ind(T) + 1;
predClass   = vec2ind(Ybin) + 1;

disp('Справжні класи:');
disp(targetClass);

disp('Передбачені класи:');
disp(predClass);

%% Побудова простору входів і областей класифікації
x1 = linspace(min(P(1,:)) - 0.5, max(P(1,:)) + 0.5, 200);
x2 = linspace(min(P(2,:)) - 0.5, max(P(2,:)) + 0.5, 200);
[X1, X2] = meshgrid(x1, x2);

gridPoints = [X1(:)'; X2(:)'];
Ygrid = net(gridPoints);
YgridBin = round(Ygrid);
classGrid = vec2ind(YgridBin) + 1;
classMap = reshape(classGrid, size(X1));

%% Графік
figure;
hold on;
grid on;
box on;

% Фон областей класифікації
contourf(X1, X2, classMap, 4, 'LineColor', 'none');
colormap(parula);
alpha(0.35);

% Межі між областями
contour(X1, X2, classMap, 'k', 'LineWidth', 1.5);

% Точки різних класів
idx1 = find(targetClass == 1); % [0;0]
idx2 = find(targetClass == 2); % [0;1]
idx3 = find(targetClass == 3); % [1;0]
idx4 = find(targetClass == 4); % [1;1]

plot(P(1, idx1), P(2, idx1), 'ko', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'k');
plot(P(1, idx2), P(2, idx2), 'bs', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'b');
plot(P(1, idx3), P(2, idx3), 'r^', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'r');
plot(P(1, idx4), P(2, idx4), 'gd', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'g');

xlabel('x_1');
ylabel('x_2');
title('Класифікація двохелементних образів на чотири класи');
legend('[0;0]', '[0;1]', '[1;0]', '[1;1]', 'Location', 'best');
hold off;
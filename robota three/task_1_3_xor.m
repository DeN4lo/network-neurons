X = [0 1 0 1;   % X(1)
     0 0 1 1];  % X(2)
Y = [0 1 1 0];  % Y = XOR(X1, X2)

% Топологія: 1 прихований шар з 3 нейронами
% Евристика: (кількість входів + виходів) / 2 = (2+1)/2 ≈ 2, беремо 3
net = feedforwardnet(3, 'trainlm');

fprintf('Архітектура XOR мережі:\n');
fprintf('  Входів:              2\n');
fprintf('  Нейронів прихованих: 3\n');
fprintf('  Виходів:             1\n\n');


%% Завдання 2. Навчання мережі

net.trainParam.epochs = 1000;
net.trainParam.goal   = 1e-5;
net.trainParam.show   = 50;

net.divideParam.trainRatio = 0.70;
net.divideParam.valRatio   = 0.15;
net.divideParam.testRatio  = 0.15;

fprintf('Запуск навчання...\n');
[net, tr] = train(net, X, Y);
fprintf('Навчання завершено.\n\n');

% Графік навчання
figure;
plotperform(tr);
title('XOR: Крива навчання (MSE vs Епохи)');


%% Завдання 3. Тестування мережі

Y_pred   = net(X);
Y_round  = round(Y_pred);

fprintf('Таблиця результатів XOR:\n');
fprintf('X1  X2 | Еталон | Вихід мережі | Округлено\n');
fprintf('--------|--------|--------------|----------\n');
for i = 1:4
    fprintf(' %d   %d  |   %d    |    %.5f   |     %d\n', ...
        X(1,i), X(2,i), Y(i), Y_pred(i), Y_round(i));
end

accuracy = mean(Y_round == Y) * 100;
fprintf('\nТочність: %.1f%%\n', accuracy);

if accuracy == 100
    fprintf('Мережа правильно моделює XOR.\n');
else
    fprintf('Є помилки — спробуйте перезапустити або збільшити нейрони.\n');
end

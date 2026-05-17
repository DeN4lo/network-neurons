%% Завдання 1. Формування навчальних даних та RBF мережі
x1 = -3 : 0.5 : 3;
x2 = -2 : 0.5 : 2;
[X1, X2] = meshgrid(x1, x2);

% Цільова функція
F_func = @(x1, x2) 2 .* exp(2 - 5 .* x2.^2) + 5 .* (x1 - x2.^2);

F_train = F_func(X1, X2);

% Масиви для newrb: вхід [2 x Q], вихід [1 x Q]
X_rbf = [X1(:)' ; X2(:)'];
T_rbf = F_train(:)';

fprintf('Навчальна вибірка:\n');
fprintf('  Розмір сітки x1: %d точок\n', length(x1));
fprintf('  Розмір сітки x2: %d точок\n', length(x2));
fprintf('  Всього точок Q = %d\n\n', size(X_rbf, 2));

% Перша мережа з goal = 0.01
fprintf('Формування RBF мережі (goal=0.01, spread=1.0)...\n');
net_rbf = newrb(X_rbf, T_rbf, 0.01, 1.0);

n_neurons = net_rbf.layers{1}.size;
fprintf('Кількість нейронів у радіальному шарі: %d\n', n_neurons);

% Збереження мережі для наступних завдань
save('net_rbf.mat', 'net_rbf', 'F_func');
fprintf('Мережу збережено у net_rbf.mat\n');


%% Завдання 2. Залежність кількості нейронів від точності

goal_values = [0.1, 0.01, 0.001];
neuron_counts = zeros(1, 3);

fprintf('\n--- Залежність нейронів від goal ---\n');
fprintf('%-8s | %-12s\n', 'goal', 'Нейронів');
fprintf('---------|-------------\n');

figure('Name', 'Кількість нейронів vs goal');

for g = 1:3
    % newrbe — без графіка, швидше
    net_tmp = newrbe(X_rbf, T_rbf, 1.0);
    % Для реального підрахунку використаємо newrb
    net_tmp2 = newrb(X_rbf, T_rbf, goal_values(g), 1.0, size(X_rbf,2), 1000);
    neuron_counts(g) = net_tmp2.layers{1}.size;
    fprintf('%-8.3f | %-12d\n', goal_values(g), neuron_counts(g));
end

% Графік
semilogx(goal_values, neuron_counts, 'b-o', 'LineWidth', 2, 'MarkerSize', 10, ...
         'MarkerFaceColor', 'b');
set(gca, 'XDir', 'reverse');
xlabel('Цільова помилка (goal), логарифмічна шкала');
ylabel('Кількість нейронів у RBF-шарі');
title('Залежність кількості нейронів від заданої точності');
grid on;
xticks(goal_values);
xticklabels({'0.1', '0.01', '0.001'});

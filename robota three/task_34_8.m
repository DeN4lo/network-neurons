%% Підготовка даних
addpath(pwd);   % якщо prprob.m лежить у цій же папці

[alphabet, targets] = prprob;

fprintf('Дані завантажено:\n');
fprintf('  Вхідний масив alphabet: %dx%d\n', size(alphabet));
fprintf('  Масив цілей targets:    %dx%d\n\n', size(targets));

% Перетворення one-hot → індекси класів (для перевірки)
Tc = vec2ind(targets);
fprintf('Приклад перших 5 класів: ');
fprintf('%d ', Tc(1:5));
fprintf('(A B C D E)\n\n');

%% Формування імовірнісної мережі

spread = 0.1;   % значення за замовчуванням для newpnn

fprintf('Формування PNN (spread = %.2f)...\n', spread);
net_pnn = newpnn(alphabet, targets, spread);
fprintf('PNN сформовано.\n\n');

fprintf('Архітектура PNN:\n');
fprintf('  Прихований шар (RBF):   %d нейронів\n', net_pnn.layers{1}.size);
fprintf('  Вихідний шар (compete): %d нейронів\n', net_pnn.layers{2}.size);

%% Тестування на чистих даних

out_pnn  = net_pnn(alphabet);
pred_idx = vec2ind(out_pnn);
true_idx = vec2ind(targets);
accuracy = mean(pred_idx == true_idx) * 100;

fprintf('\nРезультати на чистих даних:\n');
fprintf('  Правильно: %d / 26\n', sum(pred_idx == true_idx));
fprintf('  Точність:  %.1f%%\n', accuracy);

% Таблиця відповідей
fprintf('\nДетальна таблиця класифікації:\n');
fprintf('Літера | Прогноз | Правильно?\n');
fprintf('-------|---------|----------\n');
letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
for i = 1:26
    mark = '  ✓';
    if pred_idx(i) ~= i
        mark = '  ✗';
    end
    fprintf('  %s    |   %s     | %s\n', letters(i), letters(pred_idx(i)), mark);
end

% Збереження для завдання 9
save('net_pnn.mat', 'net_pnn', 'alphabet', 'targets');
fprintf('\nPNN збережено у net_pnn.mat\n');
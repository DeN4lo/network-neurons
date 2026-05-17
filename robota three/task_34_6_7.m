% Завантаження результатів
if exist('rbf_results.mat', 'file')
    load('rbf_results.mat');
    fprintf('Результати завантажено з rbf_results.mat\n\n');
else
    error('Спочатку виконайте task_34_3_5_testing.m');
end


%% Завдання 6. Surf — порівняння точної функції та RBF апроксимації

figure('Name', 'RBF Апроксимація функції', 'Position', [100 100 1200 450]);

subplot(1, 3, 1);
surf(X1_te, X2_te, F_exact, 'EdgeColor', 'none');
colorbar;
xlabel('x1'); ylabel('x2'); zlabel('f(x1,x2)');
title('Точна функція f(x1,x2)');
view([-30 35]);

subplot(1, 3, 2);
surf(X1_te, X2_te, F_approx, 'EdgeColor', 'none');
colorbar;
xlabel('x1'); ylabel('x2'); zlabel('f approx');
title('RBF апроксимація');
view([-30 35]);

subplot(1, 3, 3);
% Різниця між поверхнями (накладені)
surf(X1_te, X2_te, F_exact,  'EdgeColor', 'none', 'FaceAlpha', 0.6);
hold on;
surf(X1_te, X2_te, F_approx, 'EdgeColor', 'none', 'FaceAlpha', 0.6);
hold off;
xlabel('x1'); ylabel('x2');
title('Порівняння (синій=точна, жовтий=RBF)');
legend('Точна', 'RBF', 'Location', 'best');
view([-30 35]);

sgtitle('Апроксимація функції двох змінних за допомогою RBF мережі');


%% Завдання 7. Графік поверхні помилки апроксимації

figure('Name', 'Помилка апроксимації RBF');

subplot(1, 2, 1);
surf(X1_te, X2_te, F_error, 'EdgeColor', 'none');
colormap(gca, 'hot');
colorbar;
xlabel('x1'); ylabel('x2'); zlabel('Помилка');
title('Поверхня помилки апроксимації');
view([-30 35]);

subplot(1, 2, 2);
surf(X1_te, X2_te, abs(F_error), 'EdgeColor', 'none');
colormap(gca, 'jet');
colorbar;
xlabel('x1'); ylabel('x2'); zlabel('|Помилка|');
title('Абсолютна помилка апроксимації');
view([-30 35]);

sgtitle('Аналіз помилки RBF апроксимації');

fprintf('Графіки побудовано.\n');
fprintf('  Максимальна помилка: %.6f\n', max(abs(F_error(:))));
fprintf('  Середня помилка:     %.6f\n', mean(abs(F_error(:))));

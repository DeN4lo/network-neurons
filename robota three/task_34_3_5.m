% Завантаження мережі
if exist('net_rbf.mat', 'file')
    load('net_rbf.mat', 'net_rbf', 'F_func');
    fprintf('RBF мережу завантажено з net_rbf.mat\n\n');
else
    error('Спочатку виконайте task_34_1_2_rbf_neurons.m');
end


%% Завдання 3. Тестова сітка з подвійною щільністю

x1_test = -3 : 0.25 : 3;   % крок 0.25 замість 0.5
x2_test = -2 : 0.25 : 2;

[X1_te, X2_te] = meshgrid(x1_test, x2_test);
X_test = [X1_te(:)' ; X2_te(:)'];

fprintf('Тестова вибірка:\n');
fprintf('  Розмір сітки x1: %d точок (крок 0.25)\n', length(x1_test));
fprintf('  Розмір сітки x2: %d точок (крок 0.25)\n', length(x2_test));
fprintf('  Всього точок:    %d\n\n', size(X_test, 2));


%% Завдання 4. Відгук мережі на тестових даних

F_approx_vec = net_rbf(X_test);
F_approx     = reshape(F_approx_vec, size(X1_te));

fprintf('Апроксимацію отримано.\n');
fprintf('  Розмір вихідної матриці: %dx%d\n\n', size(F_approx));


%% Завдання 5. Масив помилок апроксимації

F_exact = F_func(X1_te, X2_te);
F_error = F_exact - F_approx;
fprintf('Статистика помилок апроксимації:\n');
fprintf('  Максимальна абс. помилка: %.6f\n', max(abs(F_error(:))));
fprintf('  Середня абс. помилка:     %.6f\n', mean(abs(F_error(:))));
fprintf('  СКВ помилки:              %.6f\n', sqrt(mean(F_error(:).^2)));

% Збереження для наступних завдань
save('rbf_results.mat', 'X1_te', 'X2_te', 'F_exact', 'F_approx', 'F_error');
fprintf('\nРезультати збережено у rbf_results.mat\n');

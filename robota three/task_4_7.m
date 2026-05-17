clc;
clear;
close all;
rng default;
targets = eye(26);

% Вхідні шаблони літер (35 елементів для кожної літери)
alphabet = round(rand(35, 26));

% Приклад реального шаблону для 7-ї літери D)
D_pattern = [ ...
    1 1 1 0 0;
    1 0 0 1 0;
    1 0 0 0 1;
    1 0 0 0 1;
    1 0 0 0 1;
    1 0 0 1 0;
    1 1 1 0 0];

% Перетворення матриці 7x5 у вектор 35x1
alphabet(:,7) = D_pattern(:);

%% Відображення 7-ї літери
figure;
imagesc(reshape(alphabet(:,7), 7, 5));
colormap(gray);
axis equal tight;
title('7-й символ латинського алфавіту');

%% Створення багатошарової нейронної мережі
% 35 входів -> 30 нейронів у прихованому шарі -> 26 виходів
net_letters = feedforwardnet(30, 'trainlm');

% Використовуємо всі дані для навчання
net_letters.divideFcn = 'dividetrain';

% Функції активації
net_letters.layers{1}.transferFcn = 'tansig';
net_letters.layers{2}.transferFcn = 'logsig';

% Функція продуктивності
net_letters.performFcn = 'mse';

%% Параметри навчання
net_letters.trainParam.epochs = 1000;
net_letters.trainParam.goal   = 1e-4;
net_letters.trainParam.show   = 50;

%% Навчання мережі
net_letters = train(net_letters, alphabet, targets);

%% Перевірка на навчальних даних
Y_letters = net_letters(alphabet);

% Визначення номерів літер
[~, pred_letters] = max(Y_letters, [], 1);
[~, true_letters] = max(targets, [], 1);

% Обчислення точності
accuracy_letters = mean(pred_letters == true_letters) * 100;

disp('--- Класифікація латинських літер ---');
disp(['Точність на навчальних даних: ', num2str(accuracy_letters), ' %']);

%% =========================================================
% Дослідження стійкості до шуму
%% =========================================================

% Значення стандартного відхилення шуму
sigmaVec = 0:0.05:0.5;

% Масив середніх помилок
meanErr = zeros(size(sigmaVec));

for s = 1:length(sigmaVec)
    sigma = sigmaVec(s);

    totalErr = 0;
    samples = 0;

    % Для кожної літери
    for q = 1:size(alphabet, 2)

        % Генеруємо 10 зашумлених варіантів
        for k = 1:10

            % Генерація шуму
            noise = sigma * randn(size(alphabet,1), 1);

            % Обмеження шуму інтервалом [-1, 1]
            noise = max(min(noise, 1), -1);

            % Додавання шуму до символу
            x_noisy = alphabet(:,q) + noise;

            % Обмеження значень
            x_noisy = max(min(x_noisy, 1), -1);

            % Реакція мережі
            y = net_letters(x_noisy);

            % Евклідова помилка
            totalErr = totalErr + norm(y - targets(:,q));

            samples = samples + 1;
        end
    end

    % Середня помилка для поточного sigma
    meanErr(s) = totalErr / samples;
end

%% Побудова графіка залежності помилки від шуму
figure;
plot(sigmaVec, meanErr, 'o-', 'LineWidth', 2);
grid on;
xlabel('Інтенсивність шуму \sigma');
ylabel('Середня евклідова помилка');
title('Залежність помилки розпізнавання від інтенсивності шуму');
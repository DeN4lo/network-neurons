% Завантаження PNN
if exist('net_pnn.mat', 'file')
    load('net_pnn.mat', 'net_pnn', 'alphabet', 'targets');
    fprintf('PNN завантажено з net_pnn.mat\n\n');
else
    error('Спочатку виконайте task_34_8_pnn.m');
end

n_letters  = 26;
n_samples  = 10;
sigmas     = 0 : 0.05 : 0.5;
n_sigma    = length(sigmas);

mean_error = zeros(1, n_sigma);
accuracy   = zeros(1, n_sigma);

fprintf('Дослідження стійкості PNN до шуму:\n');
fprintf('%-8s | %-18s | %-12s\n', 'sigma', 'Сер. помилка', 'Точність %');
fprintf('---------|--------------------|--------------\n');

for s_idx = 1:n_sigma
    sigma     = sigmas(s_idx);
    total_err = 0;
    correct   = 0;

    for letter = 1:n_letters
        for rep = 1:n_samples

            % Шум з обрізанням до інтервалу [-1, 1]
            noise       = sigma * randn(35, 1);
            noise       = max(-1, min(1, noise));
            noisy_input = alphabet(:, letter) + noise;

            % Відгук PNN
            out = net_pnn(noisy_input);

            % Евклідова норма різниці з еталоном
            total_err = total_err + norm(out - targets(:, letter));

            % Точність класифікації
            pred = vec2ind(out);
            if pred == letter
                correct = correct + 1;
            end

        end
    end

    mean_error(s_idx) = total_err / (n_letters * n_samples);
    accuracy(s_idx)   = correct / (n_letters * n_samples) * 100;

    fprintf('%-8.2f | %-18.4f | %-12.1f\n', sigma, mean_error(s_idx), accuracy(s_idx));
end


%% Графіки

figure('Name', 'PNN: Стійкість до шуму', 'Position', [100 100 900 600]);

subplot(2, 1, 1);
plot(sigmas, mean_error, 'r-s', 'LineWidth', 2, 'MarkerSize', 8, ...
     'MarkerFaceColor', 'r');
xlabel('Стандартне відхилення шуму (\sigma)');
ylabel('Середня помилка (норма Евкліда)');
title('PNN: Залежність помилки від інтенсивності шуму');
grid on;
xlim([0 0.5]);

subplot(2, 1, 2);
plot(sigmas, accuracy, 'g-^', 'LineWidth', 2, 'MarkerSize', 8, ...
     'MarkerFaceColor', 'g');
xlabel('Стандартне відхилення шуму (\sigma)');
ylabel('Точність класифікації (%)');
title('PNN: Залежність точності від інтенсивності шуму');
ylim([0 110]);
yticks(0:10:110);
grid on;
xlim([0 0.5]);

sgtitle('Аналіз стійкості PNN до шуму (Probabilistic Neural Network)');


%% Демонстрація: класифікація зашумленої літери
figure('Name', 'Приклад зашумлених літер');
ex_letter = 1;  % Буква A
ex_sigmas = [0, 0.1, 0.2, 0.3, 0.4, 0.5];
letters   = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

for k = 1:6
    noise       = ex_sigmas(k) * randn(35, 1);
    noise       = max(-1, min(1, noise));
    noisy_a     = alphabet(:, ex_letter) + noise;

    out  = net_pnn(noisy_a);
    pred = vec2ind(out);

    subplot(2, 3, k);
    plotchar(noisy_a);
    title(sprintf('\\sigma=%.1f  →  %s', ex_sigmas(k), letters(pred)));
end
sgtitle('Класифікація зашумленої літери A при різних рівнях шуму (PNN)');

fprintf('\nПідсумок:\n');
fprintf('  sigma = 0.00 -> точність = %.1f%%\n', accuracy(1));
fprintf('  sigma = 0.25 -> точність = %.1f%%\n', accuracy(6));
fprintf('  sigma = 0.50 -> точність = %.1f%%\n', accuracy(end));

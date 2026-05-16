targets = eye(26); 
alphabet = round(rand(35, 26)); 
G_pattern = [1 0 0 0 1; 1 0 0 0 0; 1 0 0 0 0; 1 0 0 1 1; 1 0 0 0 1; 0 1 1 1 0; 0 0 0 0 0]';
alphabet(:, 7) = G_pattern(:);
net = feedforwardnet(31, 'trainlm'); 
net.divideFcn = '';
net.trainParam.goal = 1e-6;
net = train(net, alphabet, targets);
disp('Мережа успішно навчена на замінних даних!');
noise_levels = 0:0.05:0.5;
mean_errors = zeros(size(noise_levels));
for i = 1:length(noise_levels)
    std_dev = noise_levels(i);
    total_error = 0;
    
    for char_idx = 1:26
        for trial = 1:10
            noisy_input = alphabet(:, char_idx) + std_dev * randn(35, 1);
            y_out = net(noisy_input);
            total_error = total_error + norm(targets(:, char_idx) - y_out);
        end
    end
    mean_errors(i) = total_error / (26 * 10);
end

figure;
plot(noise_levels, mean_errors, 'r-s', 'LineWidth', 2);
grid on;
xlabel('Рівень шуму (std dev)');
ylabel('Середня помилка (Euclidean norm)');
title('Стійкість розпізнавання алфавіту до зашумлення');
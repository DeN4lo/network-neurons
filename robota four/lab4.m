clear; clc; close all;

%% ============================================================
%  ЗАВДАННЯ 1: 48 двоелементних векторів, 6 кластерів
%% ============================================================
fprintf('=== ЗАВДАННЯ 1 ===\n');

clusters1 = 6;
points1   = 8;   % 8 * 6 = 48
std_dev1  = 0.05;

% Фіксовані центри 6 кластерів
rng(42);
centers1 = [0.15 0.15;
            0.15 0.85;
            0.50 0.15;
            0.50 0.85;
            0.85 0.15;
            0.85 0.85]';   % 2 x 6

X1 = zeros(2, clusters1 * points1);
for k = 1:clusters1
    idx = (k-1)*points1 + (1:points1);
    X1(:, idx) = centers1(:,k) + std_dev1 * randn(2, points1);
end
fprintf('Сформовано %d векторів у %d кластерах.\n', size(X1,2), clusters1);

% --- Графік 1 ---
figure('Name','Завдання 1');
colors = lines(clusters1);
for k = 1:clusters1
    idx = (k-1)*points1 + (1:points1);
    scatter(X1(1,idx), X1(2,idx), 70, colors(k,:), 'filled'); hold on;
end
xlabel('x_1'); ylabel('x_2');
title('Завдання 1: 48 двоелементних векторів (6 кластерів)');
legend(arrayfun(@(k) sprintf('Кластер %d',k), 1:clusters1, ...
    'UniformOutput',false), 'Location','best');
grid on;

%% ============================================================
%  ЗАВДАННЯ 2: Шар Кохонена (competitive layer)
%% ============================================================
fprintf('\n=== ЗАВДАННЯ 2 ===\n');

net2 = competlayer(clusters1, 0.01, 0.001);
net2.trainParam.showWindow = false;
fprintf('Шар Кохонена сформовано (%d нейронів).\n', clusters1);

%% ============================================================
%  ЗАВДАННЯ 3: Навчання з відображенням кожні 5 епох
%% ============================================================
fprintf('\n=== ЗАВДАННЯ 3 ===\n');

total_epochs = 200;
step_epochs  = 5;

net3 = init(net2);
net3.trainParam.showWindow = false;
net3.trainParam.epochs     = step_epochs;

figure('Name','Завдання 3 — Еволюція нейронів');

for ep = step_epochs : step_epochs : total_epochs
    net3.trainParam.epochs = step_epochs;
    net3 = train(net3, X1);

    clf;
    plot(X1(1,:), X1(2,:), 'g.', 'MarkerSize', 10); hold on;
    W = net3.IW{1,1};           % clusters1 x 2
    plot(W(:,1), W(:,2), 'r*', 'MarkerSize', 14, 'LineWidth', 2);
    xlabel('x_1'); ylabel('x_2');
    title(sprintf('Завдання 3: Позиції нейронів після %d епох', ep));
    legend('Вхідні дані','Нейрони','Location','best');
    grid on;
    drawnow;
end
fprintf('Навчання шару Кохонена завершено.\n');

%% ============================================================
%  ЗАВДАННЯ 4: Класифікація нових векторів
%% ============================================================
fprintf('\n=== ЗАВДАННЯ 4 ===\n');

rng(99);
W_final = net3.IW{1,1}';    % 2 x clusters1
X_test  = zeros(2, clusters1*3);
for k = 1:clusters1
    idx = (k-1)*3 + (1:3);
    X_test(:,idx) = W_final(:,k) + std_dev1 * randn(2,3);
end

outputs4 = sim(net3, X_test);
classes4 = vec2ind(outputs4);

fprintf('Результати кластеризації тестових векторів:\n');
for i = 1:size(X_test,2)
    fprintf('  [%.3f, %.3f] -> Кластер %d\n', ...
        X_test(1,i), X_test(2,i), classes4(i));
end

figure('Name','Завдання 4');
scatter(X1(1,:), X1(2,:), 25, 'k', '.'); hold on;
scatter(X_test(1,:), X_test(2,:), 100, classes4, 'filled', ...
    'MarkerEdgeColor','k');
colormap(lines(clusters1)); colorbar;
xlabel('x_1'); ylabel('x_2');
title('Завдання 4: Кластери тестових векторів');
legend('Навчальні дані','Тестові','Location','best');
grid on;

%% ============================================================
%  ЗАВДАННЯ 5: Підбір топології SOM для 2D даних
%  УВАГА: randtop НЕ підтримується plotsomnd -> лише gridtop/hextop
%% ============================================================
fprintf('\n=== ЗАВДАННЯ 5 ===\n');

topologies5 = {'gridtop', 'hextop'};   % randtop прибрано!
sizes5      = {[2 3], [3 3], [3 4], [2 4]};

best_err5  = Inf;
best_conf5 = struct('topology','','size',[0 0],'net',[],'qe',Inf);

fprintf('%-10s  %-8s  %-14s\n', 'Топологія', 'Розмір', 'Квант.помилка');
fprintf('%s\n', repmat('-',1,36));

for ti = 1:length(topologies5)
    top = topologies5{ti};
    for si = 1:length(sizes5)
        sz = sizes5{si};
        net_s = selforgmap(sz, 100, 3, top, 'linkdist');
        net_s.trainParam.showWindow = false;
        net_s.trainParam.epochs     = 500;
        net_s = train(net_s, X1);

        out = sim(net_s, X1);
        W   = net_s.IW{1,1};
        win = vec2ind(out);
        d   = X1 - W(win,:)';
        qe  = mean(sqrt(sum(d.^2,1)));

        fprintf('%-10s  [%dx%d]     %.6f\n', top, sz(1), sz(2), qe);

        if qe < best_err5
            best_err5  = qe;
            best_conf5 = struct('topology',top,'size',sz, ...
                                'net',net_s,'qe',qe);
        end
    end
end

fprintf('\n>>> Найкраща: %s [%dx%d], QE=%.6f\n', ...
    best_conf5.topology, best_conf5.size(1), best_conf5.size(2), best_conf5.qe);

figure('Name','Завдання 5 — SOM Weight Positions');
plotsompos(best_conf5.net, X1);
title(sprintf('Завдання 5: SOM (%s, %dx%d)', ...
    best_conf5.topology, best_conf5.size(1), best_conf5.size(2)));

figure('Name','Завдання 5 — SOM Neighbor Distances');
plotsomnd(best_conf5.net);
title('Завдання 5: SOM Neighbor Distances');

%% ============================================================
%  ЗАВДАННЯ 6: 24 трьохелементних вектора, 4 кластери
%  Діапазон [0..10], std = 0.8
%% ============================================================
fprintf('\n=== ЗАВДАННЯ 6 ===\n');

clusters6 = 4;
points6   = 6;    % 4 * 6 = 24
std_dev6  = 0.8;

rng(7);
centers6 = [2.5 2.5 2.5;
            2.5 7.5 7.5;
            7.5 2.5 7.5;
            7.5 7.5 2.5]';    % 3 x 4

X6 = zeros(3, clusters6 * points6);
for k = 1:clusters6
    idx = (k-1)*points6 + (1:points6);
    X6(:,idx) = centers6(:,k) + std_dev6 * randn(3, points6);
end
fprintf('Сформовано %d тривимірних векторів у %d кластерах.\n', ...
    size(X6,2), clusters6);

figure('Name','Завдання 6');
colors6 = lines(clusters6);
for k = 1:clusters6
    idx = (k-1)*points6 + (1:points6);
    scatter3(X6(1,idx), X6(2,idx), X6(3,idx), 100, colors6(k,:), 'filled');
    hold on;
end
xlabel('x_1'); ylabel('x_2'); zlabel('x_3');
title('Завдання 6: 24 трьохелементних вектора (4 кластери)');
legend(arrayfun(@(k) sprintf('Кластер %d',k), 1:clusters6, ...
    'UniformOutput',false), 'Location','best');
grid on; view(45, 30);

%% ============================================================
%  ЗАВДАННЯ 7: Підбір SOM для 3D даних
%  Тільки gridtop / hextop (randtop не підтримується plotsomnd)
%% ============================================================
fprintf('\n=== ЗАВДАННЯ 7 ===\n');

topologies7 = {'gridtop', 'hextop'};
sizes7      = {[2 2], [2 3], [3 3], [4 4]};

best_err7  = Inf;
best_conf7 = struct('topology','','size',[0 0],'net',[],'qe',Inf);

fprintf('%-10s  %-8s  %-14s\n', 'Топологія', 'Розмір', 'Квант.помилка');
fprintf('%s\n', repmat('-',1,36));

for ti = 1:length(topologies7)
    top = topologies7{ti};
    for si = 1:length(sizes7)
        sz = sizes7{si};
        net_s7 = selforgmap(sz, 100, 3, top, 'linkdist');
        net_s7.trainParam.showWindow = false;
        net_s7.trainParam.epochs     = 500;
        net_s7 = train(net_s7, X6);

        out7 = sim(net_s7, X6);
        W7   = net_s7.IW{1,1};
        win7 = vec2ind(out7);
        d7   = X6 - W7(win7,:)';
        qe7  = mean(sqrt(sum(d7.^2,1)));

        fprintf('%-10s  [%dx%d]     %.6f\n', top, sz(1), sz(2), qe7);

        if qe7 < best_err7
            best_err7  = qe7;
            best_conf7 = struct('topology',top,'size',sz, ...
                                'net',net_s7,'qe',qe7);
        end
    end
end

fprintf('\n>>> Найкраща (3D): %s [%dx%d], QE=%.6f\n', ...
    best_conf7.topology, best_conf7.size(1), best_conf7.size(2), best_conf7.qe);

figure('Name','Завдання 7 — SOM Neighbor Distances (3D)');
plotsomnd(best_conf7.net);
title(sprintf('Завдання 7: SOM (%s, %dx%d)', ...
    best_conf7.topology, best_conf7.size(1), best_conf7.size(2)));

figure('Name','Завдання 7 — SOM Sample Hits (3D)');
plotsomhits(best_conf7.net, X6);
title('Завдання 7: SOM Sample Hits');

fprintf('\n=== ВИКОНАННЯ ЗАВЕРШЕНО ===\n');

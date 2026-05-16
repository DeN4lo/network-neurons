clc;
clear;
close all;

time = 0:0.02:2.5;

x = sin(2*pi*time);
y = 2*x + 3;

X = con2seq(x);
T = con2seq(y);

net = linearlayer(1:2, 0.01);

[Xs, Xi, Ai, Ts] = preparets(net, X, T);

net.trainParam.epochs = 1000;
net.trainParam.goal = 0.001;

net = train(net, Xs, Ts, Xi, Ai);

Y = sim(net, Xs, Xi);

Y1 = cell2mat(Y);
T1 = cell2mat(Ts);

error = T1 - Y1;

figure;
plot(error);
grid on;
xlabel('Номер вимірювання');
ylabel('Помилка');
title('Помилка на виході мережі');

figure;
plot(T1, 'b');
hold on;
plot(Y1, 'r--');
grid on;
xlabel('Номер вимірювання');
ylabel('Значення');
title('Порівняння цільового та отриманого сигналу');
legend('Цільовий сигнал', 'Вихід мережі');
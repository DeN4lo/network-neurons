clc;
clear;
close all;

X = {5 4 3 2};
T = {10 20 30 40};

net = linearlayer(1:2, 0.01);

[Xs, Xi, Ai, Ts] = preparets(net, X, T);

net.trainParam.epochs = 1000;
net.trainParam.goal = 0.001;

net = train(net, Xs, Ts, Xi, Ai);

Y = sim(net, Xs, Xi);

disp('Вихід мережі:');
disp(Y);

e = gsubtract(Ts, Y);
mse_error = mse(e);

disp('Середньоквадратична помилка:');
disp(mse_error);
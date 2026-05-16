
X = [0 1 0 1; 0 0 1 1]; 
T = [0 1 1 0];          

net = feedforwardnet(3, 'trainlm');

net.trainParam.epochs = 1000; 
net.trainParam.goal = 1e-5;    

net.divideFcn = '';
net = train(net, X, T);

Y = sim(net, X);

disp('Вхідні комбінації:');
disp(X);
disp('Результат роботи нейромережі:');
disp(Y);
disp('Округлені логічні значення:');
disp(round(Y));

result = net([0; 0]);
result = round(result);
disp('0 xor 0 = ');
disp(result);
data2 = load('C:\Users\majac\Desktop\Machine learning coursera\exercice 2\ex2\ex2data2.txt');
min(data2);
max(data2);
data2;
X1 = data2(:, 1); %X1 = [ones(m, 1) X1];
%size(X1)
X2= data2(:, 2); %X2 = [ones(m, 1) X2];
%size(X2)
 
y = data2(:, 3);
%plotData(data2);
newX = mapFeature(X1, X2);
[m, n] = size(newX);

%newX = [ones(m, 1) out];
% Add intercept term to x and X_test

% Initialize fitting parameters
initial_theta = zeros(n , 1);
size(newX)
size(newX(:,1))
size(newX(:,2:end));
size(initial_theta);

[J, grad, lamda] = costFunctionReg(initial_theta, newX, y, 100)
%[j1, grad1] = costFunction(initial_theta, newX, y)

[theta, cost] = ...
	fminunc(@(t)(costFunctionReg(t, newX, y, lamda)), initial_theta, options)
  
plotDecisionBoundary(data2, theta, newX, y)
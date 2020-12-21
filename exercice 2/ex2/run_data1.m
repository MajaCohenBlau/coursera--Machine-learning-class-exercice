data1 = load('C:\Users\majac\Desktop\Machine learning coursera\exercice 2\ex2\ex2data1.txt');
size(data1)

%plotData(data1)

%% cost function


%sigmoid(y)

X = data1(:, [1, 2]); y = data1(:, 3);
[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
[J, grad] = costFunction(initial_theta, X, y)
test_theta = [-24; 0.2; 0.2];
%[cost, grad] = costFunction(test_theta, X, y);
options = optimset('Display', 'iter','MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);


fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('Expected cost (approx): 0.203\n');
fprintf('theta: \n');
fprintf(' %f \n', theta);
fprintf('Expected theta (approx):\n');
fprintf(' 25.161\n -0.206\n -0.201\n');

plotDecisionBoundary(data1, theta, X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')

ex = [1 45 85];
accepted(ex, theta)
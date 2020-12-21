load('ex3data1.mat');
num_labels = 10;
lambda = 0.1;
%[J grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t)

theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
size(X_t)
X_t
y_t = ([1;0;1;0;1] >= 0.5);
y_t
lambda_t = 3;
##[J , grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);
##[nJ, ngrad] = ApgadalrCostFunction(theta_t, X_t, y_t, lambda_t);
##fprintf('\nApgada Cost is: %f\n', nJ);
##fprintf('\nCost is: %f\n', J);
##fprintf('Expected cost: 2.534819\n');
##fprintf('Gradients:\n');
##fprintf(' %f \n', grad);
##fprintf('Apgada Gradients:\n');
##fprintf(' %f \n', ngrad);
##fprintf('Expected gradients:\n');
##fprintf(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');
%all_theta(:,1:10)
##[all_theta] = oneVsAll(X, y, num_labels, lambda);
##all_theta(:,1:10)
####[Apgada_all_theta] = Apgada_oneVsAll(X, y, num_labels, lambda);
##Apgada_all_theta(:,1:10)
##x_pred = X(1:50, :);
##y_pred = y(1:50, :)
##size(x_pred)
##p = predictOneVsAll(all_theta, x_pred);

load('ex3weights')
size(Theta1)
size(Theta2)
pred = predict(Theta1, Theta2, X)

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


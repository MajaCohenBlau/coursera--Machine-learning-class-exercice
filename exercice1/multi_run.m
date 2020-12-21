multi_data = load('ex1data2.txt')



%% feature scaling and set train and test
scaled_data = feature_scaling(multi_data)
scaled_train = scaled_data([1:35],:)
scaled_test = scaled_data([36:47], :)
%% gradientdescent
[thetas_list, theta, iter_list ] = TheGradientDescent(scaled_train, alpha=0.07, 100);

%% plot the prediction?
length(scaled_test)
y_test_s = scaled_test(:,3)
X_test_s = [ones(12,1), scaled_test(:,[1,2])];
y_pred_s = X_test_s * theta

plot(y_test_s, y_pred_s, 'rx', 'MarkerSize', 10)
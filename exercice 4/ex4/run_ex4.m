load('ex4data1.mat');
m = size(X, 1);
nn_params = [Theta1(:) ; Theta2(:)];
lambda = 0;
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10; 

%J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   %num_labels, X, y, 1);
##J_apgada = Apgada_nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
##                   num_labels, X, y, lambda);
##
fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n(this value should be about 0.383770)\n'], J);     

lambda = 1;
%checkNNGradients;
W1 = randInitializeWeights(input_layer_size, hidden_layer_size);
W2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_theta = [W1(:) ; W2(:)];
options = optimset('GradObj', 'on', 'MaxIter', 400);
[nn_paramsNew, cost]  = fmincg(@(t)(nnCostFunction(t,input_layer_size, hidden_layer_size, ...
                  num_labels, X, y, lambda)), nn_params = initial_theta, options);
                  
                   
##costFunction = @(p) nnCostFunction(p, ...
##                                   input_layer_size, ...
##                                   hidden_layer_size, ...
##                                   num_labels, X, y, lambda);
##
##% Now, costFunction is a function that takes in only one argument (the
##% neural network parameters)
##[nn_params, cost] = fmincg(costFunction, initial_nn_params, options)
##% Also output the costFunction debugging values
##debug_J  = nnCostFunction(nn_params, input_layer_size, ...
##                          hidden_layer_size, num_labels, X, y, lambda);
##
##fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f ' ...
##         '\n(for lambda = 3, this value should be about 0.576051)\n\n'], lambda, debug_J);
##
##fprintf('Program paused. Press enter to continue.\n');
##pause;     
##         
##fprintf('Apgada Cost at parameters (loaded from ex4weights): %f ',J_apgada ) 
##
##z=10000000
##g = sigmoidGradient(z)
##
##W1 = randInitializeWeights(input_layer_size, hidden_layer_size)
##W2 = randInitializeWeights(hidden_layer_size, num_labels)
function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
a1 = [ones(m, 1) X];
%printf("the size of X in PREDICT is: %d,%d \n",size(X))
z1 = a1*Theta1';
a2 = sigmoid(z1);
a2 = [ones(size(a1, 1),1) a2];
%printf("the size of a2 in PREDICT is: %d,%d \n",size(a2))
z2 = a2 * Theta2';
H = sigmoid(z2);
[prob, P] = max(H,[],2); 
%printf("the size of H in PREDICT is: %d,%d \n",size(H))
%printf("the size of a2 in PREDICT is: %d,%d \n",size(a2))
y_Vec = (1:num_labels)==y;
%printf("the size of y_Vec in PREDICT is: %d,%d \n",size(y_Vec))
term1 = (-y_Vec).* log(H) ;
%term1 = log(H)' *(-y_Vec) ;
%printf("the size of term1 in PREDICT is: %d,%d \n",size(term1))
%term2 = log(1-H')* (1.-y_Vec);
term2 = (1.-y_Vec).* log(1-H);
%printf("the size of term2 in PREDICT is: %d,%d \n",size(term2))
reg = sum(sum(Theta1).^2) + sum(sum(Theta2).^2); 
reg_term = (lambda/(2*m))*reg;
%printf("the size of reg_term in PREDICT is: %d,%d \n",size(reg_term))
J = (1/m)*sum(sum(term1 - term2)); 
%J = (1/m)*sum(sum(term1 - term2)) + reg_term
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.
% ====================== YOUR CODE HERE ======================
for i=1:m
  ## 1. calculate hyposthesis giving thetas
    a1 = X(i, :);
    a1 = [ones(size(a1, 1),1) a1];
    %printf("the size of a1 in PREDICT is: %d,%d \n",size(a1))
    z2 = a1*Theta1';
    %printf("the size of z2 in PREDICT is: %d,%d \n",size(z2))
    a2 = sigmoid(z2);
    a2_1 = [ones(size(a1, 1),1) a2];
    %printf("the size of a2 in PREDICT is: %d,%d \n",size(a2))
    z3 = a2_1 * Theta2';
    %printf("the size of z3 in PREDICT is: %d,%d \n",size(z3))
    H = sigmoid(z3);
    %printf("the size of H in PREDICT is: %d,%d \n",size(H))
  ## 2. hypothesis versus reality
    yVector = (1:num_labels)'==y(i);
    %printf("the size of yVector in PREDICT is: %d,%d \n",size(yVector))
    delta3 = H .- yVector';
    %printf("the size of y(i, :) in PREDICT is: %d,%d \n",size(y(i, :)))
    %printf("the size of delta3 in PREDICT is: %d,%d \n",size(delta3))
    %printf("the size of Theta2 in PREDICT is: %d,%d \n",size(Theta2))
  ## 3. delta hidden layer
    sgZ2 = [ones(size(z2, 1),1) sigmoidGradient(z2)];
    %printf("the size of sgZ2 in PREDICT is: %d,%d \n",size(sgZ2))
    delta2 = (delta3 * Theta2) .* sgZ2;
    %printf("the size of delta2 in PREDICT is: %d,%d \n",size(delta2))
  ## 4. accumulate gradients for each layer
    Theta1_grad = Theta1_grad + delta2(2:end)' * a1;
##    reg1 = (lambda/m) .* Theta1;
##    Theta1_grad(:,1) = Theta1_grad(:,1) + delta2(:,1)' * a1(:,1);
##    printf("the size of Theta1_grad(:,2:end) in PREDICT is: %d,%d \n",size(Theta1_grad(:,2:end)))
##    printf("the size of delta2(3:end) in PREDICT is: %d,%d \n",size(delta2(3:end)))
##    printf("the size of a1(:,2:end) in PREDICT is: %d,%d \n",size(a1(:,2:end)))
##    printf("the size of reg1 in PREDICT is: %d,%d \n",size(reg1))
##    Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + delta2(2:end)' * a1(:,2:end) .+ reg1(:,2:end);
    Theta2_grad = Theta2_grad + delta3' * a2_1;
##    reg2 = (lambda/m) .* Theta2;
##    Theta2_grad(:,1) = Theta2_grad(:,1) + delta3(:,1)' * a2_1(:,1);
##    Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + delta3' * a2_1(:,2:end) + reg2(:,2:end);
    
  Theta1_grad_reg_term = (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)]; % 25 x 401
  Theta2_grad_reg_term = (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)]; % 10 x 26
  
  %Adding regularization term to earlier calculated Theta_grad
  Theta1_grad = Theta1_grad + Theta1_grad_reg_term;
  Theta2_grad = Theta2_grad + Theta2_grad_reg_term;
##    Theta1_grad = (1/m) * (DELTA2' * A1); % 25 x 401
##    Theta2_grad = (1/m) * (DELTA3' * A2); % 10 x 26
##    deltaL =  deltaL .+ delta2(2:end) * a2'
    %printf("the size of Theta1_grad in PREDICT is: %d,%d \n",size(Theta1_grad))
    %printf("the size of Theta2_grad in PREDICT is: %d,%d \n",size(Theta2_grad))
  
  
endfor

## calculate DELTA2
Theta1_grad = (1/m)*Theta1_grad;
Theta2_grad = (1/m)*Theta2_grad;
grad = [Theta1_grad(:) ; Theta2_grad(:)]

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

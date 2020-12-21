function [J, grad, lamda] = costFunctionReg(theta, X, y, lamda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

H = sigmoid(X*theta);
size(H);
size(y);
%a = y * log(H) + (1-y)* log(1-H)
regterm = lamda/2*m * sum(theta(2:end))^2
J = 1/m .*(-y' * log(H) -  (1-y') * log(1-H)) +   regterm; 
grad(1)= (1/m) * (X(:,1)'* (H - y));
grad(2:end) = (1/m) * (X(:,2:end)'* (H - y)) + (lamda/m).*theta(2:end);
%grad = (1/m)* (X'*(h_x-y))




% =============================================================

end

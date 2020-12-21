function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
grad_reg_term = zeros(size(theta));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
reg_term = (lambda/(2*m))*sum(theta(2:end).^2);

H =  X * theta; 
J = (1/(2*m))*sum((H - y).^2) + reg_term;
grad_reg_term = (lambda/m) .* theta(2:end);
##grad(1) = (1/m) * (H - y)'* X(:,1); 
##grad(2:end) = (1/m) * (H - y)'* X  + grad_reg_term;
grad(1) = (1/m)*(X(:,1)'*(H-y)); % scalar == 1x1
grad(2:end) = (1/m)*(X(:,2:end)'*(H-y)) + (lambda/m)*theta(2:end)











% =========================================================================

grad = grad(:);

end

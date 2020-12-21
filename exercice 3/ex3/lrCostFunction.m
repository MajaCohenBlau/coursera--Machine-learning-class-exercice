function [J, grad] = lrCostFunction(theta, Xx, Yy, lamda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(Yy); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
## printf("the size of Xx in lrCostFunction is: %d,%d \n",size(Xx))
##printf("the size of theta in lrCostFunction is: %d,%d \n",size(theta))
z = Xx*theta;
H = sigmoid(z);
##printf("the size of H in lrCostFunction is: %d,%d \n",size(H))
##printf("the size of Yy in lrCostFunction is: %d,%d \n",size(Yy))
%a = y * log(H) + (1-y)* log(1-H)
a = lamda/(2*m);
b = sum(theta(2:end).^2);
regterm = a* b;
 
M = 1/m;
##printf("the size of m in lrCostFunction is: %d,%d \n",size(m))
D = -Yy' * log(H);
##printf("the size of D in lrCostFunction is: %d,%d \n",size(D))
E = (1.-Yy') * log(1-H);
##printf("the size of E in lrCostFunction is: %d,%d \n",size(E))
C = sum((-Yy' * log(H) -  (1.-Yy') * log(1-H)));
##printf("the size of C in lrCostFunction is: %d,%d \n",size(C))
J = M .* C+ regterm; 
##printf("the size of cost J in lrCostFunction is: %d,%d \n",size(J))
%J = (1/m)*sum((-y.*log(h_x))-((1-y).*log(1-h_x))) + reg_term; % scalar
grad(1)= M * (Xx(:,1)'* (H - Yy));
grad(2:end) = M * (Xx(:,2:end)'* (H - Yy)) .+ (lamda*M).*(theta(2:end));



  
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%










% =============================================================

grad = grad(:);

end

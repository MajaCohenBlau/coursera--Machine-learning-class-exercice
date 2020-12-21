function [thetas_list, theta, iter_list, Jrow ] = TheGradientDescent(data, alpha, num_iters)
  J = 0;
  thetas_list = [];
  iter_list=[];
  m = length(data);
  n = size(data)(2);
  m
  n
  y = data(:,n);
  X = [ones(m,1), data(:,[1:n-1])];
  J_list = []
  %printf("the size of X is: %d,%d \n",size(X));
  theta = zeros(n,1);
  %printf("the size of theta is: %d,%d \n",size(theta));
  for iteration = 1:num_iters
  iter_list = [iter_list, iteration];
  cost_error =  X*theta-y;
  %printf("the size of cost_error is: %d,%d \n\n",size(cost_error));
  %cost_error
  der_error = X'*cost_error;
  %printf("the size of der_error is: %d,%d \n\n",size(der_error));
  
  theta = theta .- alpha./m.*der_error;
  %printf("Theta is: %d \n\n",theta);
  J = 1/(2.*m).*sum((X*theta-y).^2);
  jrow = [iteration, J]
  thetas_list = [thetas_list; theta(1,:), theta(2,:), theta(3,:)];
  J_list=[J_list; jrow];
  end
  
  %j_thetas
  %final_J = J_list(2, num_iters);
  
  iter = J_list(:,1)
  j=J_list(:,2)
  plot(j,iter, 'rx', 'MarkerSize', 10)
  %length(iter)
  
  %subplot(1,2,2)
  
  %printf("theta is : %d \n\n",theta)
  %printf("the final value of J is : %i \n\n",J_list(final_J))
  end
  %printf("J_list is : %f \n\n",J_list)
  %size(J_list)
  %xj = J_list'(:,1);
  %yj = J_list'(:,2);
  
  %figure
  %plot(xj, yj)
  
  
  %PlotTheData(load('ex1data1.txt'), theta(1,:), theta(2,:))


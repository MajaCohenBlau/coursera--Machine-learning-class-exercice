function J = Cost_funtion(theta0, theta1)
  data = load('ex1data1.txt')
  m = length(data);
  n = size(data)(2);
  y = data(:,2);
  X = data(:,1);
  printf("the size of X is: %d,%d \n",size(X));
  theta = zeros(n,1);
  printf("the size of theta is: %d,%d \n\n",size(theta));
  error = theta1.*X - y .+ theta0;
  %error =  X*theta-y;
  J = 1/(2.*m).*sum(error.^2);
  printf("j is: %d \n",J);
end

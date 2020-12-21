function J = ComputeTheCost(data, theta)
  m = length(data);
  n = size(data)(2);
  y = data(:,2);
  X = [ones(m,1), data(:,1)];    
  error =  X*theta-y;
  J = 1/(2.*m).*sum(error.^2);
  printf("j is: %d \n",J);
  end

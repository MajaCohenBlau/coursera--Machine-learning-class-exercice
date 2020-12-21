data = load('ex1data1.txt')

[thetas_list, theta, iter_list ] = TheGradientDescent(data, alpha=0.005, 10000)
theta
Cost_function(data,theta(1), theta(2))
xc = thetas_list(:,1);
yc = thetas_list(:,2);
%zc = ComputeTheCost(data, thetas_list)
zc = Cost_function(xc, yc);

contour(xc, yc, zc)
%Cost_function(data, 1,2)
%PlotTheData(data, theta(1,:), theta(2,:)) 


 
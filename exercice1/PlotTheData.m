% plot the data
function PlotTheData(data, theta0, theta1)
  figure
  x = data(:,1);
  y = data(:,2);
  plot(x,y,'rx', 'MarkerSize', 10);
  ylabel('Profit in $10,000s'); 
  xlabel('Population of City in 10,000s');
  yr = theta0 + theta1*x;
  hold on 
  plot(x, yr)
  end

xc = thetas_list(:,1);
  length(xc)
  yc = thetas_list(:,2);
  length(yc)
  zc = thetas_list(:,3);
  length(zc)
  subplot(1,2,1)
  surf(xc, yc, zc)



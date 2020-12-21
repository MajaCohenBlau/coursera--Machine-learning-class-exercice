##load('ex6data1.mat');
##plotData(X, y);
####C = 0.1;
##model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
##visualizeBoundaryLinear(X, y, model);



##x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
##sim = gaussianKernel(x1, x2, sigma);

##fprintf(['Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f :' ...
##         '\n\t%f\n(for sigma = 2, this value should be about 0.324652)\n'], sigma, sim);
         

load('ex6data3.mat');
plotData(X, y);

##[C, sigma] = dataset3Params(X, y, Xval, yval);
C, sigma
C1 = 9.4; sigma1 = 1.5
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);

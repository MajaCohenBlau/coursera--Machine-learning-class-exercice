function plotData(data1)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
pos = (data1(:,3)==1);
data_pos = data1(pos,:);
plot(data_pos(:,1), data_pos(:,2), 'g+', 20);

hold on

neg = (data1(:,3)==0);
data_neg = data1(neg,:);
plot(data_neg(:,1), data_neg(:,2), 'ro', 20);
axis([-1, 1.5, -1, 1.5])
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')






% =========================================================================



hold off;

end

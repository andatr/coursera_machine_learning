function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
	%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
	%regression with multiple variables
	%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
	%   cost of using theta as the parameter for linear regression to fit the 
	%   data points in X and y. Returns the cost in J and the gradient in grad

	m = length(y); % number of training examples
	predictions = X * theta;
	errors = predictions - y;
	sqrErrors = errors .^ 2;
	reg = lambda / 2 / m * (sum(theta(2:end, :) .^ 2));
	J = 1 / 2 / m * sum(sqrErrors) + reg;
	grad = 1 / m * X' * errors;
	grad(2:end, :) += lambda / m * theta(2:end, :);
end

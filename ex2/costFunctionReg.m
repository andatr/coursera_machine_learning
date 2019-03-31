function [J, grad] = costFunctionReg(theta, X, y, lambda)
	%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
	%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
	%   theta as the parameter for regularized logistic regression and the
	%   gradient of the cost w.r.t. to the parameters. 

	m = length(y); % number of training examples
	s = sigmoid(X * theta);
	J = -1 / m * (y' * log(s) + (1 - y)' * log(1 - s)) + lambda / (2 * m) * sum(theta(2:end) .^ 2);
	grad = 1 / m * X' * (s - y) + [[[0]]; lambda / m * theta(2:end)];
end

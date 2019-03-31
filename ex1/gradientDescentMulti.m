function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
	%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
	%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
	%   taking num_iters gradient steps with learning rate alpha

	m = length(y); % number of training examples
	J_history = zeros(num_iters, 1);
	for iter = 1 : num_iters
		derivative = zeros(size(theta, 1), size(theta, 2));
		for s = 1 : m
			sample = X(s, :)';
			derivative = derivative + (theta' * sample - y(s)) * sample;
		end
		theta = theta - alpha * derivative / m;
		J_history(iter) = computeCostMulti(X, y, theta);
	end
end

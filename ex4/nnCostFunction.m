function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
	%NNCOSTFUNCTION Implements the neural network cost function for a two layer
	%neural network which performs classification
	%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
	%   X, y, lambda) computes the cost and gradient of the neural network. The
	%   parameters for the neural network are "unrolled" into the vector
	%   nn_params and need to be converted back into the weight matrices. 
	% 
	%   The returned parameter grad should be a "unrolled" vector of the
	%   partial derivatives of the neural network.
	%

	m = size(X, 1);
	Y = (1 : num_labels) == y;
	
	% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
	% for our 2 layer neural network
	Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, input_layer_size + 1);
	Theta2 = reshape(nn_params(1 + hidden_layer_size * (input_layer_size + 1):end), num_labels, hidden_layer_size + 1);
	
	% regularization
	reg = lambda / 2 / m * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));
	grad_reg1 = lambda / m * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
	grad_reg2 = lambda / m * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
	
	% Forward Propagation
	A1 = [ones(m, 1) X];
	Z2 = A1 * Theta1';
	A2 = [ones(size(Z2, 1), 1) sigmoid(Z2)];
	Z3 = A2 * Theta2';
	H  = sigmoid(Z3);
	J = -1 / m * sum(sum(Y .* log(H) + (1 - Y) .* log(1 - H))) + reg;
  
	% Back Propagation
	Delta3 = H - Y;
	Delta2 = (Delta3 * Theta2) .* [ones(size(Z2, 1), 1) sigmoidGradient(Z2)];
	Delta2 = Delta2(:, 2:end);
	Theta1_grad = 1 / m * Delta2' * A1 + grad_reg1;
	Theta2_grad = 1 / m * Delta3' * A2 + grad_reg2;

	% Unroll gradients
	grad = [Theta1_grad(:) ; Theta2_grad(:)];
end

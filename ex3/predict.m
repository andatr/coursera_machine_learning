function p = predict(Theta1, Theta2, X)
	%PREDICT Predict the label of an input given a trained neural network
	%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
	%   trained weights of a neural network (Theta1, Theta2)

	A1 = [ones(size(X, 1), 1) X];
	A2 = sigmoid(A1 * Theta1');
	A2 = [ones(size(A2, 1), 1) A2];
	[class, iclass] = max(sigmoid(A2 * Theta2'), [], 2);
	p = iclass;
end

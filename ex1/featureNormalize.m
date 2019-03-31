function [X_norm, mu, sigma] = featureNormalize(X)
	%FEATURENORMALIZE Normalizes the features in X 
	%   FEATURENORMALIZE(X) returns a normalized version of X where
	%   the mean value of each feature is 0 and the standard deviation
	%   is 1. This is often a good preprocessing step to do when
	%   working with learning algorithms.

	X_norm = X;
	n = size(X, 2); % number of features
	mu = zeros(1, n);
	sigma = zeros(1, n);
	for i = 1 : n
		mu(1, i) = mean(X(:, i));
		sigma(1, i) = std(X(:, i));
		X_norm(:, i) = (X(:, i) - mu(1, i)) / sigma(1, i);
	end
end

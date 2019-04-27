function centroids = computeCentroids(X, idx, K)
	%COMPUTECENTROIDS returns the new centroids by computing the means of the 
	%data points assigned to each centroid.
	%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
	%   computing the means of the data points assigned to each centroid. It is
	%   given a dataset X where each row is a single data point, a vector
	%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
	%   example, and K, the number of centroids. You should return a matrix
	%   centroids, where each row of centroids is the mean of the data points
	%   assigned to it.

	[m n] = size(X);
	couns = zeros(K, 1);
	centroids = zeros(K, n);
	for i = 1 : m
		c = idx(i);
		centroids(c, 1:end) += X(i, 1:end);
		couns(c) += 1;
	end
	centroids = centroids ./ couns;	
end


function [C, sigma] = dataset3Params(X, y, Xval, yval)
	%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
	%where you select the optimal (C, sigma) learning parameters to use for SVM
	%with RBF kernel
	%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
	%   sigma. You should complete this function to return the optimal C and 
	%   sigma based on a cross-validation set.
	%

	C = 1;
	sigma = 0.3;
	bestErr = size(X) + 1;
	
	for Cval = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
		for sigmaVal = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
			model = svmTrain(X, y, Cval, @(x1, x2) gaussianKernel(x1, x2, sigmaVal));
			predVal = svmPredict(model, Xval);
			errVal = mean(double(predVal ~= yval));
			if (errVal < bestErr)
				bestErr = errVal;
				C = Cval;
				sigma = sigmaVal;
			end
		end
	end
end

function [Kinv]=calculateInvCovarianceMatrix(X, lengthScale, scaleFactor)
%% calculate the covariance (kernel) matrix inverse of samples X
% X is a m * n matrix of m n-dimensional points

K = scaleFactor.* exp(-squareform(pdist(X,'mahalanobis', lengthScale * lengthScale * eye(size(X,2))))./2);
K = max(K',K);
Kinv = pinv(K);
end

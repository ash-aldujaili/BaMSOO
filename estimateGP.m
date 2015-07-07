function [mu,sigma]=estimateGP(x,X,y,Kinv, lengthScale, scaleFactor)
% calculate the posterior mean and variance
k = scaleFactor .* exp(- pdist2(x,X,'mahalanobis', lengthScale * lengthScale * eye(length(x)) )./2);
k1= scaleFactor .* exp(- pdist2(x,x,'mahalanobis', lengthScale * lengthScale * eye(length(x)) )./2);
mu = k * Kinv * y;
sigma = sqrt(k1 - k * Kinv * k');


end

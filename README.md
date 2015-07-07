# :sparkles: BaMSOO :sparkles:
_Bayesian Multi-Scale Optimistic Optimization_

This is a simple MATLAB implementation of BaMSOO, described in the paper Bayesian Multi-Scale Optimistic Optimization by Wang et al.. 
The paper can be found (here)[http://jmlr.org/proceedings/papers/v33/wang14d.pdf]


The repository has four files:
* BaMSOO.m : implements the technique.
* calculateInvCovarianceMatrix.m : calculate the kernel matrix
* estimateGP.m : estimates the mean and variance
* calculateBounds.m : calculates the function bounds for a function
* runDemo.m : a simple graphical demo on a 1-D function visualizing the tree

# Demo
To run a simple demo, fire MATLAB and run the following command:
~~~
runDemo
~~~

# Optimization Problem:
To solve an optimization problem, we need to specify the function, target function value, the dimensions, the min and max range of the search space, and the evaluation budget.
An example is provided here:
~~~
% Specify the problem
func = @(x) sum((x-0.6).^2);
dimension = 2;
maxRange = 1;
minRange = -1;
numEvaluations = 1000;
ftarget = 1e-5;
% Solve the problem
[yBest, xBest, nodes ]= BaMSOO(func, ftarget, dimension, maxRange, minRange, numEvaluations);
fprintf('optimal f-value is %f\n',yBest)
~~~
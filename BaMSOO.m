function [yBest, xBest, nodes ]= BaMSOO(func, ftarget, dimension, maxRange, minRange, numEvaluations)
%% BaMSOO : Bayesian Multi-Scale Optimistic Optimization
% a tool for optimizing multi-dimensional bound-constrained black-box global optimization
% problems.
% INPUTS:
% 	func : function handler
%   ftarget : terminating function value
%   dimension : problem dimensionality
%   maxRange : max values of the search space
%   minRange : min values of the search space
%   numEvaluations : evaluation budget
%   By
%      Abdullah Al-Dujaili, Nanyang Technological University



%%%%%%%%%%%%%%%%%%%%%%
% Setting  parameters:
%%%%%%%%%%%%%%%%%%%%%%
% function-related
opts.ftarget = ftarget;% ftarget
opts.fitnessFunc = func; % spherical function
% algorithm-related
opts.NUM_DIM = dimension;
opts.MAX_FES = numEvaluations; % maximum number of evaluations
opts.MAX_RANGE = maxRange * ones(opts.NUM_DIM,1);
opts.MIN_RANGE = minRange * ones(opts.NUM_DIM,1);
opts.NUM_FOLDS = 3;
opts.MIDDLE_CELL = ceil(opts.NUM_FOLDS/2);
opts.IS_EVEN_FOLDS = (mod(opts.NUM_FOLDS,2) == 0);
opts.INV_NUM_FOLDS = 1/opts.NUM_FOLDS;
opts.maxDepthFunc = ceil(sqrt(opts.MAX_FES));% or floor(10 * log(opts.MAX_FES)^(1.5)); 
% (it is supposed to be a function of number of expansion, set to a constant here according to the paper "bandits attach function optimization"
% add a condition if the full tree is explored and 
opts.MAX_NUM_NODES = floor((opts.NUM_FOLDS^(opts.maxDepthFunc+2)-1)/(opts.NUM_FOLDS-1)); % (k^(n+1)-1)/(k-1)
opts.EVAL_GAP = 5; % the difference between numevaluation and number of nodes
% visualization-related
opts.showRect = true;
opts.IS_VERBOSE    = (opts.NUM_DIM == 2) && opts.showRect;


%%%%%%%%%%%%%%%%%%%%%%%
% start the algorithm:
%%%%%%%%%%%%%%%%%%%%%%%
% root root:
root.bc.minX = opts.MIN_RANGE;
root.bc.maxX = opts.MAX_RANGE;
root.x = mean([root.bc.minX root.bc.maxX],2);
root.y = opts.fitnessFunc(root.x);
% update best
opts.xBest = root.x;
opts.yBest = root.y;
% update counters 
opts.numFEs = 1;
opts.minDepth=1;
opts.maxDepth=1;
opts.numNodes = 1;
% GP stuffs
sizeX = 3*opts.NUM_DIM;
X = zeros(sizeX,opts.NUM_DIM);
y = zeros(sizeX,1);
X(1,:) = root.x';
y(1) = root.y;
Xidx= 1;
% N represent how many items are there in the covariance matrix
%N = 1; 
N = sizeX;
% invk counter
covCount = 0;
covRate = ceil(sizeX/2); % how often the covariance matrix is updated
% for the covariance matrix calculation, we used the squared
%Squared Exponential Kernel: the Radial Basis Function kernel, the Gaussian kernel. It has the form: 
% kSE(x,x′)=σ2exp(−(x−x′)/2ℓ2) with two parameters
% The lengthscale ℓ determines the length of the 'wiggles' in your function. In general, you won't be able to extrapolate more than ℓ units away from your data.
% The output variance σ2 determines the average distance of your function away from its mean. Every kernel has this parameter out in front; it's just a scale factor.
lengthScale = 0.01; % one third the search space should be but this is what we have set
scaleFactor = 0.01;
nodes{1} = root;






% work till you run out of FEs
while(true)
	h = opts.minDepth - 1;
	v = inf;
	while (h <= min(opts.maxDepthFunc,opts.maxDepth))
		h = h + 1;
		[~,idx] = min([nodes{h}.y]);
		node = nodes{h}(idx);
		% expand or skip:
        if(isempty(idx))
			if (h == min(opts.maxDepthFunc,opts.maxDepth)), break; end
			continue;
		end
		if (v > node.y)
			v = node.y;
		else 
			if (h == min(opts.maxDepthFunc,opts.maxDepth)), break; end
			continue;
		end	
		nodes{h}(idx)=[];
		% check if empty:
		if (isempty(nodes{h}) && (h == opts.minDepth))
			opts.minDepth = opts.minDepth + 1;
		end
		% expand, split dimension-wise
		d = mod(h - 1, opts.NUM_DIM) + 1;
		r = (node.bc.maxX(d) - node.bc.minX(d)) * opts.INV_NUM_FOLDS;
		newNodes = repmat(node, 1,opts.NUM_FOLDS);
		opts.maxDepth = max(opts.maxDepth, h+1);
		for f = 1 : opts.NUM_FOLDS % children process
			isEvaluated = false;
			opts.numNodes = opts.numNodes + 1;
			newNodes(f).bc.minX = node.bc.minX;
			newNodes(f).bc.maxX = node.bc.maxX;
			newNodes(f).bc.minX(d) = newNodes(f).bc.minX(d) + r * (f-1);
			newNodes(f).bc.maxX(d) = newNodes(f).bc.minX(d) + r;
			newNodes(f).x = mean([newNodes(f).bc.minX newNodes(f).bc.maxX],2);
			if (f == opts.MIDDLE_CELL && ~opts.IS_EVEN_FOLDS)
				newNodes(f).x = node.x;
				newNodes(f).y = node.y;
			else
				newNodes(f).x = mean([newNodes(f).bc.minX newNodes(f).bc.maxX],2);
				% node evaluation (SOO for the first sizeX items)
				if (opts.numFEs < sizeX)
					newNodes(f).y = opts.fitnessFunc(newNodes(f).x);
					isEvaluated = true;
					% updates GP stuffs:
					X(Xidx+1,:) = newNodes(f).x';
					y(Xidx+1) = newNodes(f).y;
					Xidx = mod(Xidx + 1,sizeX);
				else
					%N = N + 1; % update of this value should be reconsidered as matrix update is not always happening
					% update the covariance matrix once whole of the covariance matrix is updated
					if (covCount == 0) 
						[Kinv]=calculateInvCovarianceMatrix(X, lengthScale, scaleFactor);
					end
					covCount = mod(covCount+1, covRate);
					% calculate the bounds
					[mu,sigma]=estimateGP(newNodes(f).x',X,y,Kinv, lengthScale, scaleFactor);
					[UCB,LCB]=calculateBounds(mu,sigma, N);
					if (LCB <= opts.yBest || (opts.numNodes - opts.numFEs)> opts.EVAL_GAP)
						newNodes(f).y = opts.fitnessFunc(newNodes(f).x); 
						isEvaluated = true;
						% updates GP stuffs:
						X(Xidx+1,:) = newNodes(f).x';
						y(Xidx+1) =newNodes(f).y;
						Xidx = mod(Xidx + 1,sizeX);
					else   
						newNodes(f).y = UCB;
					end       
				end  
			end
			% plot
			if (opts.IS_VERBOSE)
				figure(1)
				xlim([opts.MIN_RANGE(1) opts.MAX_RANGE(1)]);
				ylim([opts.MIN_RANGE(2) opts.MAX_RANGE(2)])
				scatter(newNodes(f).x(1),newNodes(f).x(2),'.k')
				rectangle('Position',[newNodes(f).bc.minX(1), newNodes(f).bc.minX(2), newNodes(f).bc.maxX(1)-newNodes(f).bc.minX(1), newNodes(f).bc.maxX(2)-newNodes(f).bc.minX(2)]);
				hold on
				pause(0.1);
			end
			% check fitness value & update
			if ((opts.yBest > newNodes(f).y) && (isEvaluated))
				opts.yBest = newNodes(f).y;
				opts.xBest = newNodes(f).x;
				% target reached ?
				if (opts.yBest < ftarget)
					yBest =  opts.yBest;
					xBest =  opts.xBest;
					return;
				end
			end
			% check function evaluations
			if (isEvaluated)
				opts.numFEs = opts.numFEs + 1;
				if (opts.numFEs >= opts.MAX_FES)
					yBest =  opts.yBest;
					xBest =  opts.xBest;
					return;
				end
			end
			% if the tree is exhausted before the number of iterations
			if (opts.numNodes >= opts.MAX_NUM_NODES)
				disp(['The tree of depth ' num2str(opts.maxDepthFunc) ' has been fully explored with ' num2str(opts.numNodes) ' nodes!']);
				return;
			end 
		end % end children process
		 % put the children nodes
		if (length(nodes)>h)
			nodes{h+1} = [nodes{h+1} newNodes]; 
		else
			nodes{h+1} = newNodes;
		end
	end
end % end for h




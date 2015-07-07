function [UCB,LCB]=calculateBounds(mu,sigma, N)

n = 0.9;
% original paper values : n in (0,1), 2*log,
B = sqrt(0.01*log(N*N*pi*pi/(6*n)));
UCB = mu + sigma * B;
LCB = mu - sigma * B;

end

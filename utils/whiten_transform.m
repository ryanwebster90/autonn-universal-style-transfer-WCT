function [X,V,D,M] = whiten_transform(X,k)
% Fixes the covariance of X to be unit covariance,
% where the variance is computed between the vectorized
% channels of X.
%
% Copyright (C) Ryan Webster, 2018

X_sz = size(X);

X = reshape(X,[],size(X,3));
M = mean(X,1);
X = bsxfun(@minus,X,M);

C = X.'*X;
[V,D] = eigs(C,k);
d = diag(D);
d = d.^(-.5);
tmp = (V*diag(d)*V.');
X = X*tmp;

X = reshape(X,X_sz);



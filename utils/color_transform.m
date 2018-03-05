function X = color_transform(X,V,D,M)
% Fixes the covariance of X to be specified by V,D,M
% Assumes X has unit covariance (with whiten_transform)
%
% Copyright (C) Ryan Webster, 2018


X_sz = size(X);
X = reshape(X,[],size(X,3));
d = diag(D);
d = d.^(.5);
tmp = (V*diag(d)*V.');
X = X*tmp;

X = bsxfun(@plus,X,M);
X = reshape(X,X_sz);

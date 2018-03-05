function y = sym_conv(x,w,b,varargin)
% symmetric padding layer, derivative not implemented
%
% Copyright (C) Ryan Webster, 2018

if isa(x,'Layer')
  if numel(varargin) %set w and b
    opts.hasBias = false;
    opts.size = [];
    opts = vl_argparse(opts,varargin);
    if ~numel(opts.size)
      error('must provide size');
    end
      
    scale = sqrt(2 / prod(opts.size(1:3))) ;
    w = Param('value', randn(opts.size, 'single') * scale);
    
    if opts.hasBias
      b = Param('value',zeros(opts.size(4),1,'single'));
    else
      b = [];
    end
  end
  
  numInputDer = 1 + isa(w,'Layer') + isa(b,'Layer'); % count number differentiable layers
  y = Layer(@sym_conv,x,w,b);
  y.numInputDer = numInputDer;
  
elseif ~numel(varargin) %forward pass
  padsize = floor((size(w,1) - 1)*[.5 .5]); %.5 for 'both' padding
  y = padarray(x,padsize,'both','symmetric');
  y = vl_nnconv(y,w,b);
  
else %derivative not implemented yet
  error('derivative not implemented');
end
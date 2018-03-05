function [encoder,decoder] = get_vgg_autoencoder(N_pool,zero_memory)
% Returns an encoder as the convolutional part of the VGG-19 network up to 
% the Nth pooling layer, as specified by N_pool. The decoder is specified
% by "Universal style transfer via Feature Transforms" NIPS 2017.
% Additionally, the network is configured to delete all intermediate
% tensors during execution, as it is exclusively feedforward.
%
% Copyright (C) Ryan Webster, 2018


if nargin<2
  zero_memory = true;
end

tmp = load(['./../models/texture-synth-weights/vgg_normalised_conv',num2str(N_pool+1),'_1.mat']);
x = Input();

% ENCODER

% normalize input
w = single(gpuArray(tmp.conv1_weight));
w = permute(w,[3,4,2,1]);
x = vl_nnconv(x,w,single(gpuArray(tmp.conv1_bias(:))));

i = 2;
for pool = 1:N_pool
  
  if pool == 1
    N_blocks = 2;
  elseif pool >=3
    N_blocks = 3;
  else
    N_blocks = 1;
  end
  
  for block = 1:N_blocks
    w = single(gpuArray(tmp.(['conv',num2str(i),'_weight'])));
    w = permute(w,[3,4,2,1]);
    b = single(gpuArray(tmp.(['conv',num2str(i),'_bias'])));
    x = conv_layer(x,w,b);
    x = vl_nnrelu(x);
    i = i+1;
  end
  
  x = vl_nnpool(x,[2 2],'method','max','stride', [2 2],'pad',[0 1 0 1]);
  
  w = single(gpuArray(tmp.(['conv',num2str(i),'_weight'])));
  w = permute(w,[3,4,2,1]);
  b = single(gpuArray(tmp.(['conv',num2str(i),'_bias'])));
  x = conv_layer(x,w,b);
  x = vl_nnrelu(x);
  i = i+1;
  
end

x.name = 'output';

if zero_memory
  %zero memory evaluation
  layers = x.find();
  % set all layers to nonprecious before compilation
  for l = 1:numel(layers)-1
    layers{l}.precious = false;
  end
  encoder = Net(x,'conserveMemory',[true true]);
  encoder.move('gpu');
else
  encoder = Net(x,'conserveMemory',[true true]);
  encoder.move('gpu');
end


x = Input();

tmp = load(['./../models/texture-synth-weights/feature_invertor_conv',num2str(N_pool+1),'_1.mat']);

w = single(gpuArray(tmp.(['conv',num2str(i),'_weight'])));
w = permute(w,[3,4,2,1]);
b = single(gpuArray(tmp.(['conv',num2str(i),'_bias'])));
x = conv_layer(x,w,b);
x = vl_nnrelu(x);
i = i+1;
% DECODER

for pool = N_pool :-1:1
  
  x = repelem(x,2,2);
  
  if pool >=3
    N_blocks = 4;
  else
    N_blocks = 2;
  end
  
  for block = 1:N_blocks
    
    w = single(gpuArray(tmp.(['conv',num2str(i),'_weight'])));
    w = permute(w,[3,4,2,1]);
    b = single(gpuArray(tmp.(['conv',num2str(i),'_bias'])));
    x = conv_layer(x,w,b);
    x = vl_nnrelu(x);
    i = i+1;
  end
  
end

x.name = 'output';

if zero_memory
  %zero memory evaluation
  layers = x.find();
  % set all layers to nonprecious before compilation
  for l = 1:numel(layers)-1
    layers{l}.precious = false;
  end
  decoder = Net(x,'conserveMemory',[true true]);
  decoder.move('gpu');
  
else
  decoder = Net(x);
  decoder.move('gpu');
end


function x = conv_layer(x,w,b)

% x = periodic_conv(x,w,b);
x = sym_conv(x,w,b);




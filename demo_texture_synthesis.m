% synthesizes a texture according to
% "Universal style transfer via feature transforms", NIPS 2017
%
% Copyright (C) Ryan Webster, 2018

% fn = './../simoncelli textures/campbell.jpg';
fn = 'in1.jpg';

%options
N_iter = 3;
N_eig = 150;
in_scale = .25;
pool_layers = [4,3]; %pooling layers of vgg-19 for coarse to fine refinement

x0 = double(imread(fn))/255;
x0 = resize_image_2D(x0,in_scale);
x0 = single(gpuArray(x0));

y0 = rand(size(x0),'like',x0);

figure
for iter = 1:N_iter
  for pool = pool_layers
    
    % get encoder / decoder for this level of refinement
    [encoder_net,decoder_net] = get_vgg_autoencoder(pool);
    
    % get covariance of texture in feature domain
    encoder_net.eval({'input1',x0},'forward');
    z0 = encoder_net.getValue('output');
    N_eig = min(N_eig, size(z0,3)-5); % use a few less eigenvalues then the channel size
    [z0w,V,D,M] = whiten_transform(double(gather(z0)),N_eig);
    
    % fix covariance of synthesis to match texture
    encoder_net.eval({'input1',y0},'forward');
    y0 = encoder_net.getValue('output');
    y0 = whiten_transform(double(gather(y0)),N_eig);
    y0 = single(gpuArray(color_transform(y0,V,D,M)));
    
    %invert into image domain
    decoder_net.eval({'input1',y0},'forward');
    y0 = decoder_net.getValue('output');
    
  end
  
end







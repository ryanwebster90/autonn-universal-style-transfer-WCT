% feed forward style transfer found in
% "Universal style transfer via feature transforms", NIPS 2017
%
% Copyright (C) Ryan Webster, 2018

fn_content = 'bela.jpg';
% fn_style = 'flowers1.jpg';
fn_style = 'in1.jpg';

%scale for input images
content_scale = .5;
style_scale = .25;

N_eig = 250;
N_iter = 2; % number of refinement iterations
alpha = .1; % blending of original image (alpha = 1 is no style)
pool_layers = [4,3]; 


%read input images
x_style = double(imread(fn_style))/255;
x_style = resize_image_2D(x_style,style_scale);
x_style = gpuArray(single(x_style));

x_content = double(imread(fn_content))/255;
x_content = resize_image_2D(x_content,content_scale);
x_content = gpuArray(single(x_content));


figure
y = x_content;
for iter = 1:N_iter
  
  % coarse to fine refinement
  for pool = pool_layers
    
    % get encoder / decoder for this level of refinement
    [encoder_net,decoder_net] = get_vgg_autoencoder(pool);
    
    % get covariance for style image in feature domain (pool of vgg-19)
    encoder_net.eval({'input1',x_style},'forward');
    z0 = encoder_net.getValue('output');
    N_eig = min(N_eig, size(z0,3)-5); % use a few less eigenvalues then the channel size
    [z0w,V,D,M] = whiten_transform(double(gather(z0)),N_eig);
    
    % fix covariance for content image to be that of style image
    encoder_net.eval({'input1',y},'forward');
    y = encoder_net.getValue('output');
    y = whiten_transform(double(gather(y)),N_eig);
    y_style = single(gpuArray(color_transform(y,V,D,M)));
    
    % get content features
    encoder_net.eval({'input1',x_content},'forward');
    y_content = encoder_net.getValue('output');
    
    % style blending factor alpha
    y = alpha*y_content + (1-alpha)*y_style;
    
    % invert representation to image domain
    decoder_net.eval({'input1',y},'forward');
    y = decoder_net.getValue('output');
    
    % fix size being slightly off
    if pool == pool_layers(1)
      x_content = resize_image_2D(x_content,size(y));
    end
    
    imshow(y);drawnow;
  end
end







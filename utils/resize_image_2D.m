function im_out = resize_image_2D(im,new_size)
% "toolbox free" image resizing
% also handles other datatypes, casting after resizing


im0 = im;
im = gather(im);
if ndims(im) > 3 || ndims(im) <2
    error('number of dims for image must be 2 or 3');
end
if ndims(new_size)< 1 || ndims(new_size)>2
    error('newsize must be a scalar or [new_M,new_N]');
end
[m1,n1,c] = size(im);

if numel(new_size)==1
    ratio = new_size;
    m2 = ceil(m1*ratio);
    n2 = ceil(n1*ratio);
else
    m2 = new_size(1);
    n2 = new_size(2);
end

use_imtb = license('test','image_toolbox');

%use image processing toolbox by default
if use_imtb
  im_out = imresize(im,[m2,n2],'method','lanczos3');
  im_out = cast(im_out,'like',im0);
else

im_out = [];
for i = 1:c
    [Y,X] = meshgrid( (0:n1-1)/(n1-1), (0:m1-1)/(m1-1));
    [YI,XI] = meshgrid( (0:n2-1)/(n2-1), (0:m2-1)/(m2-1));

    im_out = cat(3,im_out,interp2(Y,X, squeeze(im(:,:,i)), YI,XI,'bilinear'));
end

im_out = cast(im_out,'like',im0);

end




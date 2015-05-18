function [ims,labels] = get_batch(imdb, images, varargin)
% GET_BATCH  Load, preprocess, and pack images for CNN evaluation

opts.imageSize = [320, 320] ;
opts.numAugments = 1 ;
opts.transformation = 'none' ;
opts.averageImage = [] ;
opts.rgbVariance = zeros(0,3,'single') ;
opts.labelStride = 1 ;
opts.labelOffset = 0 ;
opts.classWeights = ones(1,21,'single') ;
opts.interpolation = 'bilinear' ;
opts.numThreads = 1 ;
opts.prefetch = false ;
opts = vl_argparse(opts, varargin);

if opts.prefetch
  %vl_imreadjpeg(images, 'numThreads', opts.numThreads, 'prefetch') ;
  ims = [] ;
  labels = [] ;
  return ;
end

if ~isempty(opts.rgbVariance) && isempty(opts.averageImage)
  opts.averageImage = zeros(1,1,3) ;
end
if numel(opts.averageImage) == 3
  opts.averageImage = reshape(opts.averageImage, 1,1,3) ;
end

% space for images
ims = zeros(opts.imageSize(1), opts.imageSize(2), 3, ...
  numel(images)*opts.numAugments, 'single') ;

% space for labels
lx = opts.labelOffset : opts.labelStride : opts.imageSize(2) ;
ly = opts.labelOffset : opts.labelStride : opts.imageSize(1) ;
labels = zeros(numel(ly), numel(lx), 2, numel(images)*opts.numAugments, 'single') ;
classWeights = [0 opts.classWeights(:)'] ;

im = cell(1,numel(images)) ;
si = 1 ;
for i=1:numel(images)

  % acquire image
  if isempty(im{i})
    rgbPath = sprintf(imdb.paths.image, imdb.images.name{images(i)}) ;
    labelsPath = sprintf(imdb.paths.classSegmentation, imdb.images.name{images(i)}) ;
    %rgb = single(imread(rgbPath)) ;
    rgb = vl_imreadjpeg({rgbPath}) ;
    rgb = rgb{1} ;
    anno = imread(labelsPath) ;
  else
    rgb = im{i} ;
  end
  if size(rgb,3) == 1
    rgb = cat(3, rgb, rgb, rgb) ;
  end

  % crop & flip
  w = size(rgb,2) ;
  h = size(rgb,1) ;
  for ai = 1:opts.numAugments
    sz = opts.imageSize(1:2) ;
    tf = rand(3)-.5 ;
    dx = floor((w - sz(2)) * tf(2)) + 1 ;
    dy = floor((h - sz(1)) * tf(1)) + 1 ;
    flip = tf(3) > 0 ;

    sx = round(linspace(dx, sz(2)+dx-1, opts.imageSize(2))) ;
    sy = round(linspace(dy, sz(1)+dy-1, opts.imageSize(1))) ;
    if flip, sx = fliplr(sx) ; end

    okx = find(1 <= sx & sx <= w) ;
    oky = find(1 <= sy & sy <= h) ;

    if ~isempty(opts.averageImage)
      offset = opts.averageImage ;
      ims(oky,okx,:,si) = bsxfun(@minus, rgb(sy(oky),sx(okx),:), offset) ;
    else
      ims(oky,okx,:,si) = rgb(sy(oky),sx(okx),:) ;
    end
    tlabels = zeros(opts.imageSize(1), opts.imageSize(2), 'uint8') + 255 ;
    tlabels(oky,okx) = anno(sy(oky),sx(okx)) ;
    tlabels = single(tlabels(ly,lx)) ;
    tlabels = mod(tlabels + 1, 256) ; % 0 = ignore, 1 = bkg
    labels(:,:,1,si) = tlabels ;
    labels(:,:,2,si) = classWeights(tlabels + 1) ;
    si = si + 1 ;
  end
end

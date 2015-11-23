function info = fcnTest(varargin)

run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;

% experiment and data paths
opts.expDir = 'data/fcn32s-voc11' ;
opts.dataDir = 'data/voc11' ;
opts.modelPath = 'data/fcn32-voc11/net-epoch-50.mat' ;
opts.modelFamily = 'matconvnet' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

% experiment setup
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat') ;
opts.vocEdition = '11' ;
opts.vocAdditionalSegmentations = true ;
opts.vocAdditionalSegmentationsMergeMode = 2 ;
opts.gpus = [] ;
opts = vl_argparse(opts, varargin) ;

resPath = fullfile(opts.expDir, 'results.mat') ;
if exist(resPath)
  info = load(resPath) ;
  return ;
end

if ~isempty(opts.gpus)
  gpuDevice(opts.gpus(1))
end

% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------

% Get PASCAL VOC 11/12 segmentation dataset plus Berkeley's additional
% segmentations
if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
else
  imdb = vocSetup('dataDir', opts.dataDir, ...
    'edition', opts.vocEdition, ...
    'includeTest', false, ...
    'includeSegmentation', true, ...
    'includeDetection', false) ;
  if opts.vocAdditionalSegmentations
    imdb = vocSetupAdditionalSegmentations(...
      imdb, ...
      'dataDir', opts.dataDir, ...
      'mergeMode', opts.vocAdditionalSegmentationsMergeMode) ;
  end
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% Get validation subset
val = find(imdb.images.set == 2 & imdb.images.segmentation) ;

% Compare the validation set to the one used in the FCN paper
% valNames = sort(imdb.images.name(val)') ;
% valNames = textread('data/seg11valid.txt', '%s') ;
% valNames_ = textread('data/seg12valid-tvg.txt', '%s') ;
% assert(isequal(valNames, valNames_)) ;

% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------

switch opts.modelFamily
  case 'matconvnet'
    net = load(opts.modelPath) ;
    net = dagnn.DagNN.loadobj(net.net) ;
    net.mode = 'test' ;
    for name = {'objective', 'accuracy'}
      net.removeLayer(name) ;
    end
    net.meta.normalization.averageImage = reshape(net.meta.normalization.rgbMean,1,1,3) ;
    predVar = net.getVarIndex('prediction') ;
    inputVar = 'input' ;
    imageNeedsToBeMultiple = true ;

  case 'ModelZoo'
    net = dagnn.DagNN.loadobj(load(opts.modelPath)) ;
    net.mode = 'test' ;
    predVar = net.getVarIndex('upscore') ;
    inputVar = 'data' ;
    imageNeedsToBeMultiple = false ;

  case 'TVG'
    net = dagnn.DagNN.loadobj(load(opts.modelPath)) ;
    net.mode = 'test' ;
    predVar = net.getVarIndex('coarse') ;
    inputVar = 'data' ;
    imageNeedsToBeMultiple = false ;
end

if ~isempty(opts.gpus)
  gpuDevice(opts.gpus(1)) ;
  net.move('gpu') ;
end
net.mode = 'test' ;

% -------------------------------------------------------------------------
% Train
% -------------------------------------------------------------------------

numGpus = 0 ;
confusion = zeros(21) ;

for i = 1:numel(val)
  imId = val(i) ;
  name = imdb.images.name{imId} ;
  rgbPath = sprintf(imdb.paths.image, name) ;
  labelsPath = sprintf(imdb.paths.classSegmentation, name) ;

  % Load an image and gt segmentation
  rgb = vl_imreadjpeg({rgbPath}) ;
  rgb = rgb{1} ;
  anno = imread(labelsPath) ;
  lb = single(anno) ;
  lb = mod(lb + 1, 256) ; % 0 = ignore, 1 = bkg

  % Subtract the mean (color)
  im = bsxfun(@minus, single(rgb), net.meta.normalization.averageImage) ;

  % Soome networks requires the image to be a multiple of 32 pixels
  if imageNeedsToBeMultiple
    sz = [size(im,1), size(im,2)] ;
    sz_ = round(sz / 32)*32 ;
    im_ = imresize(im, sz_) ;
  else
    im_ = im ;
  end

  if ~isempty(opts.gpus)
    im_ = gpuArray(im_) ;
  end

  net.eval({inputVar, im_}) ;
  scores_ = gather(net.vars(predVar).value) ;
  [~,pred_] = max(scores_,[],3) ;

  if imageNeedsToBeMultiple
    pred = imresize(pred_, sz, 'method', 'nearest') ;
  else
    pred = pred_ ;
  end

  % Accumulate errors
  ok = lb > 0 ;
  confusion = confusion + accumarray([lb(ok),pred(ok)],1,[21 21]) ;

  % Plots
  if mod(i - 1,30) == 0 || i == numel(val)
    clear info ;
    [info.iu, info.miu, info.pacc, info.macc] = getAccuracies(confusion) ;
    fprintf('IU ') ;
    fprintf('%4.1f ', 100 * info.iu) ;
    fprintf('\n meanIU: %5.2f pixelAcc: %5.2f, meanAcc: %5.2f\n', ...
            100*info.miu, 100*info.pacc, 100*info.macc) ;

    figure(1) ; clf;
    imagesc(normalizeConfusion(confusion)) ;
    axis image ; set(gca,'ydir','normal') ;
    colormap(jet) ;
    drawnow ;

    % Print segmentation
    figure(100) ;clf ;
    displayImage(rgb/255, lb, pred) ;
    drawnow ;

    % Save segmentation
    imPath = fullfile(opts.expDir, [name '.png']) ;
    imwrite(pred,labelColors(),imPath,'png');
  end
end

% Save results
save(resPath, '-struct', 'info') ;

% -------------------------------------------------------------------------
function nconfusion = normalizeConfusion(confusion)
% -------------------------------------------------------------------------
% normalize confusion by row (each row contains a gt label)
nconfusion = bsxfun(@rdivide, double(confusion), double(sum(confusion,2))) ;

% -------------------------------------------------------------------------
function [IU, meanIU, pixelAccuracy, meanAccuracy] = getAccuracies(confusion)
% -------------------------------------------------------------------------
pos = sum(confusion,2) ;
res = sum(confusion,1)' ;
tp = diag(confusion) ;
IU = tp ./ max(1, pos + res - tp) ;
meanIU = mean(IU) ;
pixelAccuracy = sum(tp) / max(1,sum(confusion(:))) ;
meanAccuracy = mean(tp ./ max(1, pos)) ;

% -------------------------------------------------------------------------
function displayImage(im, lb, pred)
% -------------------------------------------------------------------------
subplot(2,2,1) ;
image(im) ;
axis image ;
title('source image') ;

subplot(2,2,2) ;
image(uint8(lb-1)) ;
axis image ;
title('ground truth')

cmap = labelColors() ;
subplot(2,2,3) ;
image(uint8(pred-1)) ;
axis image ;
title('predicted') ;

colormap(cmap) ;

% -------------------------------------------------------------------------
function cmap = labelColors()
% -------------------------------------------------------------------------
N=21;
cmap = zeros(N,3);
for i=1:N
  id = i-1; r=0;g=0;b=0;
  for j=0:7
    r = bitor(r, bitshift(bitget(id,1),7 - j));
    g = bitor(g, bitshift(bitget(id,2),7 - j));
    b = bitor(b, bitshift(bitget(id,3),7 - j));
    id = bitshift(id,-3);
  end
  cmap(i,1)=r; cmap(i,2)=g; cmap(i,3)=b;
end
cmap = cmap / 255;

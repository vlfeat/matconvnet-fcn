function fcnTest(varargin)

run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;

% experiment and data paths
opts.expDir = 'data/fcn-baseline' ;
opts.dataDir = 'data/voc12' ;
opts.modelPath = 'data/fcn-baseline-5/net-epoch-10.mat' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

% experiment setup
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat') ;
opts.vocEdition = '12' ;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------

% Get PASCAL VOC 12 segmentation dataset plus Berkeley's additional
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
    imdb = vocSetupAdditionalSegmentations(imdb, 'dataDir', opts.dataDir) ;
  end
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% Get validation subset
val = find(imdb.images.set == 2 & imdb.images.segmentation) ;

% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------

net = load(opts.modelPath) ;
net = dagnn.DagNN.loadobj(net.net) ;
net.mode = 'test' ;
for name = {'objective', 'accuracy'}
  net.removeLayer(name) ;
end

% -------------------------------------------------------------------------
% Train
% -------------------------------------------------------------------------

numGpus = 0 ;
confusion = zeros(21) ;

% Run

predVar = net.getVarIndex('prediction') ;

for i = 1:numel(val)
  imId = val(i) ;
  rgbPath = sprintf(imdb.paths.image, imdb.images.name{imId}) ;
  labelsPath = sprintf(imdb.paths.classSegmentation, imdb.images.name{imId}) ;

  rgb = vl_imreadjpeg({rgbPath}) ;
  rgb = rgb{1} ;
  anno = imread(labelsPath) ;
  lb = single(anno) ;
  lb = mod(lb + 1, 256) ; % 0 = ignore, 1 = bkg

  im = bsxfun(@minus, single(rgb), ...
              reshape(net.meta.normalization.rgbMean,1,1,3)) ;


  sz = [size(im,1), size(im,2)] ;
  sz_ = round(sz / 32)*32 ;
  im_ = imresize(im, sz_) ;

  net.eval({'input', im_}) ;
  scores_ = gather(net.vars(predVar).value) ;
  [~,pred_] = max(scores_,[],3) ;

  pred = imresize(pred_, sz, 'method', 'nearest') ;

  % sz = min(size(pred), size(lb)) ;
  %pred = pred(1:sz(1), 1:sz(2)) ;
  %pred = padarray(pred, size(lb) - size(pred), 'replicate',
  %'post') ;

  % Accumulate errors
  ok = lb > 0 ;
  confusion = confusion + accumarray([lb(ok),pred(ok)],1,[21 21]) ;

  % Plots
  if mod(i - 1,10) == 0 || i == numel(val)
    [iu, miu, pacc, macc] = getAccuracies(confusion) ;
    fprintf('IU ') ;
    fprintf('%4.1f ', 100 * iu) ;
    fprintf('meanIU: %2.f pixelAcc: %.2f, meanAcc: %.2f\n', ...
            100*miu, 100*pacc, 100*macc) ;

    figure(1) ; clf;
    imagesc(normalizeConfusion(confusion)) ;
    axis image ; set(gca,'ydir','normal') ;
    colormap(jet) ;
    drawnow ;

    % Print segmentation
    figure(100) ;clf ;
    displayImage(rgb/255, lb, pred) ;
    drawnow ;

    %   Save segmentation
    %   imname = strcat(opts.results,sprintf('/%s.png',imdb.images.name{subset(i)}));
    %   imwrite(pred,labelColors(),imname,'png');
  end
end

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

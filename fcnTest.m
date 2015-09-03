function fcnTest(varargin)

run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;

% experiment and data paths
opts.expDir = 'data/fcn-baseline' ;
opts.dataDir = 'data/voc12' ;
opts.modelPath = 'data/fcn-baseline-2/net-epoch-7.mat' ;
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

  net.eval({'input', im}) ;
  pred = gather(net.vars(end).value) ;
  [~,pred] = max(pred,[],3) ;

  sz = min(size(pred), size(lb)) ;
  pred = pred(1:sz(1), 1:sz(2)) ;
  pred = padarray(pred, size(lb) - size(pred), 'replicate', 'post') ;

  %   Save segmentation
  %   imname = strcat(opts.results,sprintf('/%s.png',imdb.images.name{subset(i)}));
  %   imwrite(pred,labelColors(),imname,'png');

  % Print segmentation
  figure(100) ;clf ;
  displayImage(rgb/255, lb, pred) ;
  drawnow ;

  % accumulate errors
  ok = lb > 0 ;
  confusion = confusion + accumarray([lb(ok),pred(ok)],1,[21 21]) ;

  if mod(i,10) == 0
    acc = getAccuracyFromConfusion(confusion) ;
     fprintf('%4.1f ', 100 * acc) ;
     fprintf(': %4.1f\n', 100 * mean(acc)) ;
     figure(1) ; clf;
     imagesc(normalizeConfusion(confusion)) ;
     axis image ; set(gca,'ydir','normal') ;
     colormap(jet) ;
     drawnow ;
  end
end

% -------------------------------------------------------------------------
function nconfusion = normalizeConfusion(confusion)
% -------------------------------------------------------------------------
% normalize confusion by row (each row contains a gt label)
nconfusion = bsxfun(@rdivide, double(confusion), double(sum(confusion,2))) ;

% -------------------------------------------------------------------------
function accuracies = getAccuracyFromConfusion(confusion)
% -------------------------------------------------------------------------
% The accuracy is: true positive / (true positive + false positive + false negative)
% which is equivalent to the following percentage:
accuracies = zeros(1,21) ;
for c = 1:21
   gtj=sum(confusion(c,:));
   resj=sum(confusion(:,c));
   gtjresj=confusion(c,c);
   accuracies(c) = double(gtjresj)/(double(gtj+resj-gtjresj)+10^-4);
end

% -------------------------------------------------------------------------
function displayImage(im, lb, pred)
% -------------------------------------------------------------------------
subplot(2,2,1) ;
image(im) ;
axis image ;

subplot(2,2,2) ;
image(uint8(lb-1)) ;
axis image ;

cmap = labelColors() ;
subplot(2,2,3) ;
image(uint8(pred-1)) ;
axis image ;

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


function cnn_seg_eval(varargin)

run ~/src/vlfeat/toolbox/vl_setup ;
run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;

opts.expDir = 'data/baseline-5' ;
opts.imdbPath = 'data/voc11/imdb.mat';
opts.modelPath = fullfile(opts.expDir, 'net-epoch-10.mat');
opts.dataDir = 'data/voc11' ;
opts.vocEdition = '11' ;
opts.numFetchThreads = 12 ;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

load(opts.modelPath, 'net') ;
net.layers(end) = [] ;
info = vl_simplenn_display(net) ;
net.normalization.offset = info.receptiveFieldOffset(end) ;
net.normalization.stride = info.receptiveFieldStride(end) ;

% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------

if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
else
  imdb = voc_init('dataDir', opts.dataDir, ...
    'edition', opts.vocEdition, ...
    'includeTest', true, ...
    'includeSegmentation', true, ...
    'includeDetection', true) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end
train = find(imdb.images.set == 1 & imdb.images.segmentation) ;
val = find(imdb.images.set == 2 & imdb.images.segmentation) ;

% -------------------------------------------------------------------------
%                                                               Evaluation
% -------------------------------------------------------------------------

evaluate(net, imdb, val) ;

% -------------------------------------------------------------------------
function evaluate(net, imdb, subset)
% -------------------------------------------------------------------------

avg = mean(mean(net.normalization.averageImage,2),1) ;

tp = zeros(20,1) ;
tn = zeros(20,1);
fp = zeros(20,1) ;
fn = zeros(20,1) ;

confusion = zeros(21) ;

for i = 1:numel(subset)
  im = single(imread(sprintf(imdb.paths.image, imdb.images.name{subset(i)}))) ;
  lb = imread(sprintf(imdb.paths.classSegmentation, imdb.images.name{subset(i)})) ;

  im_ = bsxfun(@minus, im, avg) ;
  res = vl_simplenn(net, im) ;

  [~,pred] = max(res(end).x,[],3) ;
  pred = pred - 1 ;

  t = maketform('affine',eye(3)) ;
  offset = net.normalization.offset;
  stride = net.normalization.stride;
  lx = offset + (0:size(pred,2)-1)*stride ;
  ly = offset + (0:size(pred,1)-1)*stride ;
  lbx = 1:size(lb,2) ;
  lby = 1:size(lb,1) ;

  pred_ = imtransform(pred, t, ...
    'nearest', ...
    'udata', offset + [0,size(pred,2)-1]*stride, ...
    'vdata', offset + [0,size(pred,1)-1]*stride, ...
    'xdata', [1 size(lb,2)], ...
    'ydata', [1 size(lb,1)], ...
    'size', [size(lb,1) size(lb,2)]) ;

  ok = lb < 255 ;
  confusion = confusion + accumarray([lb(ok),pred_(ok)]+1,1,[21 21]) ;

  if mod(i-1,10) == 0
    acc = getAccuracyFromConfusion(confusion) ;
    fprintf('%4.1f ', 100 * acc) ;
    fprintf(': %4.1f\n', 100 * mean(acc)) ;
    figure(1) ; clf;
    %displayImage(im, lb, pred, pred_) ;
    imagesc(normalizeConfusion(confusion)) ; axis image ; set(gca,'ydir','normal') ;
    drawnow ;
  end
end

function nconfusion = normalizeConfusion(confusion)
% normalize confusion by row (each row contains a gt label)
nconfusion = bsxfun(@rdivide, confusion, sum(confusion,2)) ;

function accuracies = getAccuracyFromConfusion(confusion)
% The accuracy is: true positive / (true positive + false positive + false negative) 
% which is equivalent to the following percentage:
for c = 1:21
   gtj=sum(confusion(c,:));
   resj=sum(confusion(:,c));
   gtjresj=confusion(c,c);
   accuracies(c) = gtjresj/(gtj+resj-gtjresj);
end

function displayImage(im, lb, pred, pred_)
figure(1) ; clf ;
subplot(2,2,1) ;
imagesc(im/255) ;
axis image ;

subplot(2,2,2) ;
imagesc(lb) ;
axis image ;

subplot(2,2,3) ;
imagesc(pred) ;
axis image ;

subplot(2,2,4) ;
imagesc(pred_) ;
  axis image ;

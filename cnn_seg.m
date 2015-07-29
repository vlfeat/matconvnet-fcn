function cnn_seg(varargin)

%run ~/src/vlfeat/toolbox/vl_setup ;
run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;

opts.expDir = 'data/baseline-vgg16-32svoc12' ;
opts.imdbPath = 'data/voc12/imdb-ext.mat' ;
opts.imdbStatsPath = fullfile(opts.expDir, 'imdb-stats.mat') ;
opts.averageIm = fullfile(opts.expDir, 'imageStats.mat') ;
opts.modelPath = 'data/trainFCN/pascalvoc2012-FCN32s' ;
opts.dataDir = 'data/voc12' ;
opts.vocEdition = '12' ;
opts.vocExtend = true ;
opts.numFetchThreads = 12 ;
opts.train.batchSize = 20 ;
opts.train.numSubBatches = 10 ;
opts.train.continue = true ;
opts.train.gpus = [ 1  ] ;
opts.train.prefetch = true ;
opts.train.expDir = opts.expDir ;
opts.train.learningRate = [0.0001*ones(1,175)] ;
opts.train.numEpochs = numel(opts.train.learningRate) ;
opts.train.confusion = zeros(21,'uint32') ; 
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

load(opts.modelPath) ;


% net.layers.addLayer('objective', dagnn.Loss('loss', 'softmaxlog'), ...
%              {'prediction','label'}, 'objective') ;
% net.layers.vars(40).precious = 1;

% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------

bopts.numThreads = opts.numFetchThreads ;
bopts.labelStride = 1 ;
bopts.labelOffset = 1 ;
bopts.classWeights = ones(1,21,'single') ; % class weight is not necessary

if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
else
  imdb = voc_init('dataDir', opts.dataDir, ...
    'edition', opts.vocEdition, ...
    'includeTest', false, ...
    'includeSegmentation', true, ...
    'includeDetection', false) ;
  if opts.vocExtend
    imdb = voc12_seg_extend(imdb, 'dataDir', 'VOC2011') ;
  end
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

train = find(imdb.images.set == 1 & imdb.images.segmentation) ;
val = find(imdb.images.set == 2 & imdb.images.segmentation) ;

if ~exist(opts.imdbStatsPath)
  classCounts = zeros(21,1) ;
  for i = 1:numel(train)
    fprintf('%s: computing stats for training image %d\n', mfilename, i) ;
    lb = imread(sprintf(imdb.paths.classSegmentation, imdb.images.name{train(i)})) ;
    ok = lb < 255 ;
    classCounts = classCounts + accumarray(lb(ok(:))+1, 1, [21 1]) ;
  end
  mkdir(fileparts(opts.imdbStatsPath)) ;
  save(opts.imdbStatsPath, 'classCounts') ;
else
  load(opts.imdbStatsPath, 'classCounts') ;
end

if ~exist(opts.averageIm)
  % compute image statistics (mean, RGB covariances etc)
  [averageImage, rgbMean, rgbCovariance] = getImageStats(imdb, bopts) ;
  save(opts.averageIm, 'averageImage', 'rgbMean', 'rgbCovariance') ;
else
  load(opts.averageIm, 'averageImage') ;
end

net.normalization.averageIm = averageImage ;
net.classes.description = imdb.classes.name ;

%bopts = net.normalization ;
bopts.averageImage = mean(mean(net.normalization.averageIm,1),2) ;

% [ims,labels] = get_batch(imdb, train(1:10), bopts) ;
% figure(100) ;clf ;
% subplot(1,2,1) ; vl_imarraysc(ims) ; axis image ;
% subplot(1,2,2) ; vl_imarray(squeeze(labels)) ; colormap([[0,0,0];jet(20);[1 1 1]]) ; axis image ; colorbar ;
% axis image ;

% -------------------------------------------------------------------------
%                                               Stochastic gradient descent
% -------------------------------------------------------------------------
opts.train.extractStatsFn = @accuracy_segmentation ;
% opts.train.errorLabels = {'pixel accuracy' ,'mean accuracy', 'mean IU'} ;

fn = getBatchWrapper( bopts) ;
info = cnn_train_dag(net.layers, imdb, fn, opts.train, ...
  'train', train, ...
  'val', val) ;

% -------------------------------------------------------------------------
function fn = getBatchWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) get_batch(imdb,batch,opts,'prefetch',nargout==0) ;

function [stats, opts] = accuracy_segmentation(net, opts)
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
sel = find(cellfun(@(x) isa(x,'dagnn.Loss'), {net.layers.block})) ;
stats = struct() ;
for i = 1:numel(sel)
  stats.(net.layers(sel(i)).name) = net.layers(sel(i)).block.average ;
end

v = net.getVarIndex('prediction') ;
predictions = gather(net.vars(v).value) ;
[~,predictions] = sort(predictions, 3, 'descend') ;

v = net.getVarIndex('label') ;
labels = gather(net.vars(v).value) ;
% Remove class weight dimension
if size(labels,3) == 2
  labels(:,:,2,:) = [] ;
end

% error = bsxfun(@eq, predictions, labels) ;
nlabels = size(predictions, 3) ;

pred = predictions(:,:,1,:) ;
ok = labels > 0 ;
opts.confusion = opts.confusion + uint32(accumarray([labels(ok),pred(ok)],1,[21 21])) ;
mp_accuracies = zeros(1,nlabels,'double') ; miu_accuracies = zeros(1,nlabels,'double') ; 
for c = 1:nlabels
  gtj = sum(opts.confusion(c,:)) ;
  resj = sum(opts.confusion(:,c)) ;
  gtjresj = opts.confusion(c,c) ;
  mp_accuracies(c) = double(gtjresj)/(double(gtj)+10^-4) ;
  miu_accuracies(c) = double(gtjresj)/(double(gtj+resj-gtjresj)+10^-4) ;
end

stats.pixel_accuracy = double(sum(sum(opts.confusion(logical(eye(nlabels))))))/double((sum(sum(opts.confusion)+10^-4))) ;
stats.mean_accuracy = 1/nlabels * (sum(mp_accuracies)) ;
stats.meanIU = 1/nlabels * (sum(miu_accuracies)) ;

% -------------------------------------------------------------------------
function [averageImage, rgbMean, rgbCovariance] = getImageStats(imdb, opts)
% -------------------------------------------------------------------------
train = find(imdb.images.set == 1 & imdb.images.segmentation) ;
bs = 20 ;
fn = getBatchWrapper(opts) ;
for t=1:bs:numel(train)
  batch_time = tic ;
  batch = train(t:min(t+bs-1, numel(train))) ;
  fprintf('collecting image stats: batch starting with image %d ...', batch(1)) ;
  temp = fn(imdb, batch) ;
  z = reshape(permute(temp{2},[3 1 2 4]),3,[]) ;
  n = size(z,2) ;
  avg{t} = mean(temp{2}, 4) ;
  rgbm1{t} = sum(z,2)/n ;
  rgbm2{t} = z*z'/n ;
  batch_time = toc(batch_time) ;
  fprintf(' %.2f s (%.1f images/s)\n', batch_time, numel(batch)/ batch_time) ;
end
averageImage = mean(cat(4,avg{:}),4) ;
rgbm1 = mean(cat(2,rgbm1{:}),2) ;
rgbm2 = mean(cat(3,rgbm2{:}),3) ;
rgbMean = rgbm1 ;
rgbCovariance = rgbm2 - rgbm1*rgbm1' ;

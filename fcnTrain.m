function fcnTrain(varargin)

run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;

opts.expDir = 'data/fcn' ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat') ;
opts.imdbStatsPath = fullfile(opts.expDir, 'imdbStats.mat') ;

opts.dataDir = 'data/voc12' ;
opts.vocEdition = '12' ;
opts.vocAdditionalSegmentations = false ;

opts.sourceModelPath = 'data/models/imagenet-vgg-verydeep-16.mat' ;

opts.numFetchThreads = 1 ;

opts.train.batchSize = 20 ;
opts.train.numSubBatches = 10 ;
opts.train.continue = true ;
opts.train.gpus = [  ] ;
opts.train.prefetch = true ;
opts.train.expDir = opts.expDir ;
opts.train.learningRate = [0.0001*ones(1,175)] ;
opts.train.numEpochs = numel(opts.train.learningRate) ;

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

% Get training and test/validation subsets
train = find(imdb.images.set == 1 & imdb.images.segmentation) ;
val = find(imdb.images.set == 2 & imdb.images.segmentation) ;

% Get dataset statistics
if exist(opts.imdbStatsPath)
  stats = load(opts.imdbStatsPath) ;
else
  stats = getDatasetStatistics(imdb) ;
  save(opts.imdbStatsPath, '-struct', 'stats') ;
end

% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------

% Get initial model from VGG-VD-16
net = fcnInitializeModel('sourceModelPath', opts.sourceModelPath) ;
net.meta.normalization.rgbMean = stats.rgbMean ;
net.meta.classes = imdb.classes.name ;

% -------------------------------------------------------------------------
% Train
% -------------------------------------------------------------------------

% Setup data fetching options
bopts.numThreads = opts.numFetchThreads ;
bopts.labelStride = 1 ;
bopts.labelOffset = 1 ;
bopts.classWeights = ones(1,21,'single') ;

% Launch SGD
info = cnn_train_dag(net, imdb, getBatchWrapper(bopts), opts.train, ...
  'train', train, ...
  'val', val) ;

% -------------------------------------------------------------------------
function fn = getBatchWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatch(imdb,batch,opts,'prefetch',nargout==0) ;



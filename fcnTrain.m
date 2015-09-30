function fcnTrain(varargin)
%FNCTRAIN Train FCN model using MatConvNet

run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;

% experiment and data paths
opts.expDir = 'data/fcn-baseline' ;
opts.dataDir = 'data/voc12' ;
opts.modelType = 'fcn32' ;
opts.sourceModelPath = 'data/models/imagenet-vgg-verydeep-16.mat' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

% experiment setup
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat') ;
opts.imdbStatsPath = fullfile(opts.expDir, 'imdbStats.mat') ;
opts.vocEdition = '12' ;
opts.vocAdditionalSegmentations = true ;

opts.numFetchThreads = 1 ; % not used yet

% training options (SGD)
opts.train.batchSize = 20 ;
opts.train.numSubBatches = 10 ;
opts.train.continue = true ;
opts.train.gpus = [] ;
opts.train.prefetch = true ;
opts.train.expDir = opts.expDir ;
opts.train.learningRate = 0.0001 * ones(1,175) ;
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
    imdb = vocSetupAdditionalSegmentations(imdb, ...
                                           'dataDir', opts.dataDir, ...
                                           'preserveValSet', true) ;
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
if any(strcmp(opts.modelType, {'fcn16s', 'fcn8s'}))
  % upgrade model to FCN16s
  net = fcnInitializeModel16s(net) ;
end
if strcmp(opts.modelType, 'fcn8s')
  % upgrade model fto FCN8s
  net = fcnInitializeModel8s(net) ;
end
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
bopts.rgbMean = stats.rgbMean ;
bopts.useGpu = numel(opts.train.gpus) > 0 ;

% Launch SGD
info = cnn_train_dag(net, imdb, getBatchWrapper(bopts), opts.train, ...
  'train', train, ...
  'val', val) ;

% -------------------------------------------------------------------------
function fn = getBatchWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatch(imdb,batch,opts,'prefetch',nargout==0) ;

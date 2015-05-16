function cnn_seg(varargin)

run ~/src/vlfeat/toolbox/vl_setup ;
run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;

opts.expDir = 'data/baseline-3' ;
opts.imdbPath = 'data/voc11/imdb.mat' ;
opts.modelPath = 'matconvnet/data/models/imagenet-vgg-f.mat' ;
opts.dataDir = 'data/voc11' ;
opts.vocEdition = '11' ;
opts.numFetchThreads = 12 ;
opts.train.batchSize = 16 ;
opts.train.numSubBatches = 1 ;
opts.train.continue = true ;
opts.train.gpus = [] ;
opts.train.prefetch = true ;
opts.train.sync = false ;
opts.train.expDir = opts.expDir ;
opts.train.learningRate = 0.01*logspace(-2, -4, 60) ;
opts.train.numEpochs = numel(opts.train.learningRate) ;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

net = load(opts.modelPath) ;
net.layers = net.layers(1:15) ;
info = vl_simplenn_display(net) ;

net.layers{end+1} = struct('type', 'conv', 'name', 'class6', ...
                           'weights', {{0.01 * randn(1, 1, 256, 256, 'single'), zeros(1,1,256,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0, ...
                           'learningRate', 10*[1 2], ...
                           'weightDecay', [1 0]) ;
net.layers{end+1} = struct('type', 'relu', 'name', 'relu6') ;
net.layers{end+1} = struct('type', 'conv', 'name', 'class7', ...
                           'weights', {{0.01 * randn(1, 1, 256, 21, 'single'), zeros(1,1,21,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0, ...
                           'learningRate', 10*[1 2], ...
                           'weightDecay', [1 0]) ;
net.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;


% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------

%bopts = net.normalization ;
bopts.numThreads = opts.numFetchThreads ;
bopts.averageImage = mean(mean(net.normalization.averageImage,1),2) ;
bopts.labelStride = info.receptiveFieldStride(1,end) ;
bopts.labelOffset = info.receptiveFieldOffset(1,end) ;

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

% [ims,labels] = get_batch(imdb, train(1:10), bopts) ;
% figure(100) ;clf ;
% subplot(1,2,1) ; vl_imarraysc(ims) ; axis image ;
% subplot(1,2,2) ; vl_imarray(squeeze(labels)) ; colormap([[0,0,0];jet(20);[1 1 1]]) ; axis image ; colorbar ;
% axis image ;

% -------------------------------------------------------------------------
%                                               Stochastic gradient descent
% -------------------------------------------------------------------------

fn = getBatchWrapper(bopts) ;
[net,info] = cnn_train(net, imdb, fn, opts.train, ...
  'train', train, ...
  'val', val, ...
  'conserveMemory', true) ;

% -------------------------------------------------------------------------
function fn = getBatchWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) get_batch(imdb,batch,opts) ;

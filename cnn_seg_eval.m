function cnn_seg_eval(varargin)

% run ~/src/vlfeat/toolbox/vl_setup ;
run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;

model = 'data/FCN8s';
opts.gpus = [  ] ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.expDir = '' ;
opts.imdbPath = 'data/voc12/imdb-ext.mat';
opts.averageIm = fullfile(opts.expDir, 'data/voc12/imageStats.mat') ;
opts.modelPath = fullfile(opts.expDir, model);
opts.dataDir = 'data/voc12' ;
opts.vocEdition = '12' ;
opts.numFetchThreads = 12 ;
opts.results = fullfile(opts.expDir,'results') ;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

load(opts.modelPath, 'net') ;
net.layers.mode = 'test';

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
% load(opts.averageIm, 'averageImage') ;
% net.normalization.averageImage = averageImage ; 
train = find(imdb.images.set == 1 & imdb.images.segmentation) ;
val = find(imdb.images.set == 2 & imdb.images.segmentation) ;


% -------------------------------------------------------------------------
%                                                       GPUs initialization
% -------------------------------------------------------------------------
numGpus = numel(opts.gpus) ;
if numGpus > 1
  if isempty(gcp('nocreate')),
    parpool('local',numGpus) ;
    spmd, gpuDevice(opts.gpus(labindex)), end
  end
  net.layers.move('gpu') ;
elseif numGpus == 1
  gpuDevice(opts.gpus)
    net.layers.move('gpu') ;
end
if exist(opts.memoryMapFile), delete(opts.memoryMapFile) ; end

% -------------------------------------------------------------------------
%                                                               Evaluation
% -------------------------------------------------------------------------
evaluate(opts, net, imdb, val) ;


% -------------------------------------------------------------------------
function evaluate(opts, net, imdb, subset)
% -------------------------------------------------------------------------
bopts.averageImage = mean(mean(net.normalization.averageImage,2),1) ;
numGpus = numel(opts.gpus) ;
confusion = zeros(21,'uint32') ;

for i = 1:numel(subset)
  im0 = single(imread(sprintf(imdb.paths.image, imdb.images.name{subset(i)}))) ;

  [im, lb] = loadImage(imdb, subset(i), bopts) ;
  
  % classify shifted images
  if numGpus > 0
   im = gpuArray(im) ;
  end
  
   net.layers.eval({'input', im}) ;
%    res = vl_simplenn(net, im, [], [],...
%      'disableDropout',true) ;
  pred = gather(net.layers.vars(end).value) ;
  [~,pred] = max(pred,[],3) ;

%   Save segmentation
%   imname = strcat(opts.results,sprintf('%s.png',imdb.images.name{subset(i)}));
%   imwrite(pred,labelColors(),imname,'png');

%   Print segmentation
%   figure(100) ;clf ;
%   displayImage(im0/255, lb(1:size(im0,1),1:size(im0,2)), pred(1:size(im0,1),1:size(im0,2))) ;
%   drawnow;
  
  ok = lb > 0 ;
  confusion = confusion + uint32(accumarray([lb(ok),pred(ok)],1,[21 21])) ;

  if mod(i,10) == 0
    acc = getAccuracyFromConfusion(confusion) ;
%     fprintf('%4.1f ', 100 * acc) ;
%     fprintf(': %4.1f\n', 100 * mean(acc)) ;
%     figure(1) ; clf;
%     %displayImage(im, lb, pred) ;
%     imagesc(normalizeConfusion(confusion)) ; axis image ; set(gca,'ydir','normal') ;
%     colormap(jet) ;
%     drawnow ;
  end
end

function [im, lb] = loadImage(imdb, imId, bopts)
% Load image and associated labels
  % Resizing image to fit the network
  imageSize = [500, 500] ;
  if numel(bopts.averageImage) == 3
    bopts.averageImage = reshape(bopts.averageImage, 1,1,3) ;
  end
    
  % acquire image
  rgbPath = sprintf(imdb.paths.image, imdb.images.name{imId}) ;
  labelsPath = sprintf(imdb.paths.classSegmentation, imdb.images.name{imId}) ;
  
  rgb = vl_imreadjpeg({rgbPath}) ;
  rgb = rgb{1} ;
  anno = imread(labelsPath) ;

  w = size(rgb,2) ; h = size(rgb,1) ; 
  im = zeros(imageSize(1), imageSize(2), 3, 'single') ;
  im(1:h,1:w,:) = bsxfun(@minus, rgb, bopts.averageImage) ;
  
  lb = zeros(imageSize(1), imageSize(2), 'single') + 255 ;
  lb(1:h,1:w) = single(anno) ;
  lb = mod(lb + 1, 256) ; % 0 = ignore, 1 = bkg


function nconfusion = normalizeConfusion(confusion)
% normalize confusion by row (each row contains a gt label)
nconfusion = bsxfun(@rdivide, confusion, sum(confusion,2)) ;

function accuracies = getAccuracyFromConfusion(confusion)
% The accuracy is: true positive / (true positive + false positive + false negative) 
% which is equivalent to the following percentage:
accuracies = zeros(1,21) ;
for c = 1:21
   gtj=sum(confusion(c,:));
   resj=sum(confusion(:,c));
   gtjresj=confusion(c,c);
   accuracies(c) = double(gtjresj)/(double(gtj+resj-gtjresj)+10^-4);
end

function displayImage(im, lb, pred)
figure(1) ; clf ;
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

function cmap = labelColors()
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

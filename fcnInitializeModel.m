function net = fcnInitializeModel(varargin)
%FCNINITIALIZEMODEL Initialize the FCN model from VGG-VD-16

opts.sourceModelUrl = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat' ;
opts.sourceModelPath = 'data/models/imagenet-vgg-verydeep-16.mat' ;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                    Load & download the source model if needed (VGG VD 16)
% -------------------------------------------------------------------------
if ~exist(opts.sourceModelPath)
  fprintf('%s: downloading %s\n', opts.sourceModelUrl) ;
  mkdir(fileparts(opts.sourceModelPath)) ;
  urlwrite('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat', opts.sourceModelPath) ;
end
net = load(opts.sourceModelPath) ;

% -------------------------------------------------------------------------
%                                  Edit the model to create the FCN version
% -------------------------------------------------------------------------

% Add dropout to the source emodel
drop1 = struct('name', 'dropout1', 'type', 'dropout', 'rate' , 0.5) ;
drop2 = struct('name', 'dropout2', 'type', 'dropout', 'rate' , 0.5) ;
net.layers = [net.layers(1:33) drop1 net.layers(34:35) drop2 net.layers(36:end)] ;

% Convert the model from SimpleNN to DagNN
net = dagnn.DagNN.fromSimpleNN(net) ;

% Add more padding to the input layer
net.layers(1).block.pad = 100 ;

% Modify the bias learning rate for all layers
for i = 1:numel(net.layers)-1
  if (isa(net.layers(i).block, 'dagnn.Conv') && net.layers(i).block.hasBias)
    filt = net.getParamIndex(net.layers(i).params{1}) ;
    bias = net.getParamIndex(net.layers(i).params{2}) ;
    net.params(bias).learningRate = 2 * net.params(filt).learningRate ;
  end
end

% Modify the last fully-connected layer to have 21 output classes
% Initialize the new filters to zero
for i = net.getParamIndex(net.layers(end-1).params) ;
  sz = size(net.params(i).value) ;
  sz(end) = 21 ;
  net.params(i).value = zeros(sz, 'single') ;
end
net.layers(end-1).block.size = size(...
  net.params(net.getParamIndex(net.layers(end-1).params{1})).value) ;

% Remove the last loss layer
net.removeLayer('loss') ;
net.layers(end).outputs = {'x38'} ;

% Add deconvolutional layer implementing bilinear interpolation
filters = single(bilinear_u(64, 21, 21)) ;
net.addLayer('deconv', ...
  dagnn.ConvTranspose('size', size(filters), ...
                      'upsample', 32, ...
                      'crop', 6, ...
                      'numGroups', 21, ...
                      'hasBias', false), ...
             'x38', 'prediction', 'deconvf') ;

f = net.getParamIndex('deconvf') ;
net.params(f).value = filters ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;

% Make the output of the bilinear interpolator is not discared for
% visualization purposes
net.vars(net.getVarIndex('prediction')).precious = 1 ;

% Add loss layer
net.addLayer('objective', ...
  SegmentationLoss('loss', 'softmaxlog'), ...
  {'prediction', 'label'}, 'objective') ;

% Add accuracy layer
net.addLayer('accuracy', ...
  SegmentationAccuracy(), ...
  {'prediction', 'label'}, 'accuracy') ;




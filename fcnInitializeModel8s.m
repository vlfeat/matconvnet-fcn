% Load net
load('FCN16s.mat') ;
net = net.layers ;

%% Remove the last layer
net.removeLayer('upsample') ;

%% Add the first Deconv layer
filters = single(bilinear_u(4, 1, 21)) ;
net.addLayer('deconv2', ...
  dagnn.ConvTranspose('size', size(filters), ...
                      'upsample', 2, ...
                      'hasBias', false), ...
             'x42', 'x43', 'deconv2f') ;
f = net.getParamIndex('deconv2f') ;
net.params(f).value = filters ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;
         
%% Add a convolutional layer that take as input the pool4 layer
net.addLayer('pool3_c', ...
     dagnn.Conv('size', [1 1 256 21]), ...
     'x17', 'x44', {'pool3_cf','pool3_cb'});
     
f = net.getParamIndex('pool3_cf') ;
net.params(f).value = zeros(1, 1, 512, 21, 'single') ;
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;

f = net.getParamIndex('pool3_cb') ;
net.params(f).value = zeros(1, 1, 21, 'single') ;
net.params(f).learningRate = 2 ;
net.params(f).weightDecay = 1 ;

net.addLayer('crop2', ...
    dagnn.Crop('crop', 66), ...
    'x44', 'x45') ;

%% Add the sumwise layer
net.addLayer('sumwise2', ...
     dagnn.SumWise(), ...
     {'x43', 'x45'}, 'x46') ; 

% Add deconvolutional layer implementing bilinear interpolation
filters = single(bilinear_u(16, 21, 21)) ;
net.addLayer('deconv', ...
  dagnn.ConvTranspose('size', size(filters), ...
                      'upsample', 8, ...
                      'crop', 18, ...
                      'numGroups', 21, ...
                      'hasBias', false), ...
             'x46', 'prediction', 'deconvf') ;

f = net.getParamIndex('deconvf') ;
net.params(f).value = filters ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;

% Make the output of the bilinear interpolator is not discared for
% visualization purposes
net.vars(net.getVarIndex('prediction')).precious = 1 ;

% % Add loss layer
% net.addLayer('objective', ...
%   SegmentationLoss('loss', 'softmaxlog'), ...
%   {'prediction', 'label'}, 'objective') ;
% 
% % Add accuracy layer
% net.addLayer('accuracy', ...
%   SegmentationAccuracy(), ...
%   {'prediction', 'label'}, 'accuracy') ;

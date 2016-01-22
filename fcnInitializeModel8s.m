function net = fcnInitializeModel8s(net)
%FCNINITIALIZEMODEL8S Initialize the FCN-8S model from FCN-16S

%% Remove the last layer
net.removeLayer('deconv16') ;

%% Add the first deconv layer
filters = single(bilinear_u(4, 1, 21)) ;
net.addLayer('deconv2bis', ...
  dagnn.ConvTranspose('size', size(filters), ...
                      'upsample', 2, ...
                      'crop', 1, ...
                      'hasBias', false), ...
             'x41', 'x42', 'deconv2bisf') ;
f = net.getParamIndex('deconv2bisf') ;
net.params(f).value = filters ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;

%% Add a convolutional layer that take as input the pool3 layer
net.addLayer('skip3', ...
     dagnn.Conv('size', [1 1 256 21]), ...
     'x17', 'x43', {'skip3f','skip3b'});

f = net.getParamIndex('skip3f') ;
net.params(f).value = zeros(1, 1, 256, 21, 'single') ;
net.params(f).learningRate = 0.01 ;
net.params(f).weightDecay = 1 ;

f = net.getParamIndex('skip3b') ;
net.params(f).value = zeros(1, 1, 21, 'single') ;
net.params(f).learningRate = 2 ;
net.params(f).weightDecay = 1 ;

%% Add the sumwise layer
net.addLayer('sum2', dagnn.Sum(), {'x42', 'x43'}, 'x44') ;

%% Add deconvolutional layer implementing bilinear interpolation
filters = single(bilinear_u(16, 21, 21)) ;
net.addLayer('deconv8', ...
  dagnn.ConvTranspose('size', size(filters), ...
                      'upsample', 8, ...
                      'crop', 4, ...
                      'numGroups', 21, ...
                      'hasBias', false, ...
                      'opts', net.meta.cudnnOpts), ...
             'x44', 'prediction', 'deconv8f') ;

f = net.getParamIndex('deconv8f') ;
net.params(f).value = filters ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;

% Make the output of the bilinear interpolator is not discared for
% visualization purposes
net.vars(net.getVarIndex('prediction')).precious = 1 ;

% empirical test
if 0
  figure(100) ; clf ;
  n = numel(net.vars) ;
  for i=1:n
    vl_tightsubplot(n,i) ;
    showRF(net, 'input', net.vars(i).name) ;
    title(sprintf('%s', net.vars(i).name)) ;
    drawnow ;
  end
end

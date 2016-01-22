function net = fcnInitializeModel16s(net)
%FCNINITIALIZEMODEL16S Initialize the FCN-16S model from FCN-32

%% Remove the last layer
net.removeLayer('deconv32') ;

%% Add the first Deconv layer
filters = single(bilinear_u(4, 1, 21)) ;
net.addLayer('deconv2', ...
  dagnn.ConvTranspose('size', size(filters), ...
                      'upsample', 2, ...
                      'crop', [1 1 1 1], ...
                      'hasBias', false), ...
             'x38', 'x39', 'deconv1f') ;
f = net.getParamIndex('deconv1f') ;
net.params(f).value = filters ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;

%% Add skip layers on top of pool4
net.addLayer('skip4', ...
     dagnn.Conv('size', [1 1 512 21], 'pad', 0), ...
     'x24', 'x40', {'skip4f','skip4b'});

f = net.getParamIndex('skip4f') ;
net.params(f).value = zeros(1, 1, 512, 21, 'single') ;
net.params(f).learningRate = 0.1 ;
net.params(f).weightDecay = 1 ;

f = net.getParamIndex('skip4b') ;
net.params(f).value = zeros(1, 1, 21, 'single') ;
net.params(f).learningRate = 2 ;
net.params(f).weightDecay = 1 ;

%% Add the summation layer
net.addLayer('sum1', dagnn.Sum(), {'x39', 'x40'}, 'x41') ;

%% Add deconvolutional layer implementing bilinear interpolation
filters = single(bilinear_u(32, 21, 21)) ;
net.addLayer('deconv16', ...
  dagnn.ConvTranspose('size', size(filters), ...
                      'upsample', 16, ...
                      'crop', 8, ...
                      'numGroups', 21, ...
                      'hasBias', false, ...
                      'opts', net.meta.cudnnOpts), ...
             'x41', 'prediction', 'deconv16f') ;

f = net.getParamIndex('deconv16f') ;
net.params(f).value = filters ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;

% Make sure that the output of the bilinear interpolator is not discared for
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

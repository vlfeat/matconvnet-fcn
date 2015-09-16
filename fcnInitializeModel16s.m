function net = fcnInitializeModel16s(net)
%FCNINITIALIZEMODEL16S Initialize the FCN-16S model from FCN-32

%% Remove the last layer
net.removeLayer('deconv32') ;

%% Add the first Deconv layer
filters = single(bilinear_u(3, 1, 21)) ;
net.addLayer('deconv2', ...
  dagnn.ConvTranspose('size', size(filters), ...
                      'upsample', 2, ...
                      'crop', [1 2 1 2], ...
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
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;

f = net.getParamIndex('skip4b') ;
net.params(f).value = zeros(1, 1, 21, 'single') ;
net.params(f).learningRate = 2 ;
net.params(f).weightDecay = 1 ;
        
%% Add the summation layer
net.addLayer('sum1', Sum(), {'x39', 'x40'}, 'x41') ; 

%% Add deconvolutional layer implementing bilinear interpolation
filters = single(bilinear_u(32, 21, 21)) ;
net.addLayer('deconv16', ...
  dagnn.ConvTranspose('size', size(filters), ...
                      'upsample', 16, ...
                      'crop', 14+15, ...
                      'numGroups', 21, ...
                      'hasBias', false), ...
             'x41', 'prediction', 'deconv16f') ;

f = net.getParamIndex('deconv16f') ;
net.params(f).value = filters ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;

% Make sure that the output of the bilinear interpolator is not discared for
% visualization purposes
net.vars(net.getVarIndex('prediction')).precious = 1 ;

% empirical test
if 1
  figure(1) ; clf ;
  subplot(2,2,1) ;
  showRF(net, 'input', 'x39') ;
  subplot(2,2,2) ;
  showRF(net, 'input', 'x40') ;
  subplot(2,2,3) ;
  showRF(net, 'input', 'x41') ;
  subplot(2,2,4) ;
  showRF(net, 'input', 'prediction') ;
end


classdef SegmentationAccuracy < dagnn.ElementWise

  properties (Transient)
    pixelAccuracy = 0
    meanAccuracy = 0
    meanIntersectionUnion = 0
    average = [0;0;0]
    confusion = 0
  end

  methods
    function outputs = forward(obj, inputs, params) 
      predictions = gather(inputs{1}) ;
      labels = gather(inputs{2}) ;
      [~,predictions] = sort(predictions, 3, 'descend') ;

      % error = bsxfun(@eq, predictions, labels) ;
      nlabels = size(predictions, 3) ;

      % compute statistics only on accumulated pixels
      ok = labels > 0 ;
      obj.confusion = obj.confusion + ...
        accumarray([labels(ok),predictions(ok)],1,[21 21]) ;
      
      % compute various statistics of the confusion matrix
      gt = sum(obj.confusion,2) ;
      res = sum(obj.confusion,1)' ;
      dg = diag(obj.confusion) ;

      obj.pixelAccuracy = sum(dg) / max(1,sum(obj.confusion(:))) ;
      obj.meanAccuracy = mean(dg ./ max(1, gt)) ;
      obj.meanIntersectionUnion = mean(dg ./ max(1, gt + res - dg)) ;      
      
      obj.average = [obj.pixelAccuracy ; obj.meanAccuracy ; obj.meanIntersectionUnion] ;
      outputs{1} = obj.average ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = [] ;
      derInputs{2} = [] ;
      derParams = {} ;
    end

    function reset(obj)
      obj.confusion = 0 ;
      obj.pixelAccuracy = 0 ;
      obj.meanAccuracy = 0 ;
      obj.meanIntersectionUnion = 0 ;
      obj.average = [0;0;0] ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1 1 1 inputSizes{1}(1)] ;
    end

    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

    function obj = SegmentationAccuracy(varargin)
      obj.load(varargin) ;
    end
  end
end

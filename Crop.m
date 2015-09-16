classdef Crop < dagnn.ElementWise
  properties
    crop = [0 0 0 0]
  end

  properties (Transient)
    inputSize = {}
  end
  
  methods
    function outputs = forward(self, inputs, params)
      outputs{1} = vl_nncrop(inputs{1}, self.crop) ;
      obj.inputSize = size(inputs{1}) ;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      derInputs{1} = vl_nncrop(inputs{1}, self.crop, derOutputs{1}, obj.inputSize) ;
      derParams = {} ;
    end

    function obj = Crop(varargin)
      obj.load(varargin) ;
    end
    
    function set.crop(obj, crop)
      if numel(crop) == 1
        obj.crop = [crop crop crop crop] ;
      elseif numel(crop) == 2
        obj.crop = crop([1 1 2 2]) ;
      else
        obj.crop = crop ;
      end
    end
    
    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes{1} = [...
        inputSizes{1}(1) - obj.crop(1) - obj.crop(2), ...
        inputSizes{1}(2) - obj.crop(3) - obj.crop(4), ...
        1, ...
        inputSizes{1}(4)] ;
    end

    function rfs = getReceptiveFields(obj)
      rfs.size = [1, 1] ;
      rfs.stride = [1, 1] ;
      rfs.offset = [1 + obj.crop(1), 1 + obj.crop(3)] ;
    end
  end
  
end

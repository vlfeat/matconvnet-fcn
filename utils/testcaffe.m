run ~/src/vlfeat/toolbox/vl_setup
run matconvnet/matlab/vl_setupnn

%model = '32s' ;
%model = '16s' ;
model = '8s' ;

blobs = load(sprintf('../testcaffe/blobs%s.mat', model)) ;
opts.modelPath = sprintf('matconvnet/data/models/pascal-fcn%s-dag.mat',model) ;

% load and fix model
net = dagnn.DagNN.loadobj(load(opts.modelPath)) ;
net.mode = 'test' ;
net.conserveMemory = false ;

filename = '/home/vedaldi/src/deep-seg/data/voc11/JPEGImages/2008_000073.jpg' ;
im = single(imread(filename)) ;
im = bsxfun(@minus, single(im), ...
            reshape(net.meta.normalization.averageImage,1,1,3)) ;

net.eval({'data', im}) ;

ok = ismember({net.vars.name}, fieldnames(blobs)) ;
names = {net.vars(ok).name} ;

for i=1:numel(names) ;
  name = names{i} ;
  matches = cellfun(@(x) ~isempty(regexp(x,['^' name 'x*$'])), {net.vars.name}) ;
  j = max(find(matches)) ;
  if isempty(j), continue ; end

  name_ = net.vars(j).name ;
  a=net.vars(j).value ;
  b=blobs.(name);
  b=permute(b, [3 4 2 1]) ;
  if i==1, b = b(:,:,[3 2 1]) ; end % BGR RGB

  del=  max(abs((a(:) - b(:)))) ;
  str=sprintf('%d %s vs %s = %g',i,name,name_,del);
  disp(str);
end

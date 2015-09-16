function showRF(net, in, out)

ii = net.getVarIndex(in) ;
oi = net.getVarIndex(out) ;

if 0
  x = randn(768, 768, 3, 1, 'single') ;
  net.reset() ;
  net.initParams() ;
  
  net.conserveMemory = true ;
  net.vars(oi).precious = true ;
  net.eval({in,x}) ;
  y = net.vars(oi).value ;
  
  dzdy = zeros(size(y), 'single') ;
  i = round((size(dzdy,1)+1)/2) ;
  j = round((size(dzdy,2)+1)/2) ;
  dzdy(i,j,:) = 100 * randn(1,1,size(dzdy,3)) ;
  
  net.eval({in,x},{out,dzdy}) ;
  map = net.vars(ii).der ;
  
  rfs = net.getVarReceptiveFields(in) ;
  rf = rfs(oi) ;
  a = rf.stride .* ([i j] - 1) + rf.offset - (rf.size-1)/2 ;
  b = rf.stride .* ([i j] - 1) + rf.offset + (rf.size-1)/2 ;
  
  cla ;
  imagesc(log(1e-12+sum(abs(map),3))) ;
  axis image ;
  hold on ;
  vl_plotbox([a-0.5, b+0.5]','g') ;  
else
  sizes = net.getVarSizes({'input', [512 512 3 1]}) ;
  rfs = net.getVarReceptiveFields(in) ;
  rf = rfs(oi) ;
  if isempty(rf.size), return ; end
  [u,v] = meshgrid(1:sizes{oi}(2),1:sizes{oi}(1));
  vp = rf.stride(1) .* (v-1) + rf.offset(1) ;
  up = rf.stride(1) .* (u-1) + rf.offset(1) ;
  cla ;
  vl_plotbox([1-.5 1-.5 512+.5 512+.5]','b') ;
  axis equal ; hold on ;
  vl_plotgrid(up,vp) ;
end
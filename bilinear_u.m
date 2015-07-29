function f = bilinear_u( k, numGroups, numclasses )
%BILINEAR_U(K, NUMGROUPS, NUMCLASSES) compute a square bilinear filter of 
% size k for deconv layer of depth numClasses and number of groups numGroups

factor = floor((k+1)/2) ;
if rem(k,2)==1
  center = factor ;
else
  center = factor + 0.5 ;
end
C = 1:k ;
if numGroups ~= numclasses
  f = zeros(k,k,numGroups,numclasses) ;
else
  f = zeros(k,k,1,numclasses) ;
end

for i =1:numclasses
  if numGroups ~= numclasses
    index = i ;
  else 
    index = 1 ;
  end
  f(:,:,index,i) = (ones(1,k) - abs(C-center)./factor)'*(ones(1,k) - abs(C-center)./(factor));
end
end


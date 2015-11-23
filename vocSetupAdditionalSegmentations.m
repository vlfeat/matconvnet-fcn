function imdb = vocSetupAdditionalSegmentations(imdb, varargin)
%VOCSETUPADDITIONALSEGMENTATIONS Download additional Berkeley segmentation data for PASCAL VOC 12
%   IMDB = VOCSETUPADDITIONALSEGMENTATIONS(IMDB) downloads and setups
%   Berkeley additional segmentations for the PASCAL VOC 2012 segmentation
%   challenge data.
%
%   Example::
%        imdb = vocSetup('dataDir', 'data/voc12') ;
%        imdb = vocSetupAdditionalSegmentations(...
%             imdb, 'dataDir', 'data/voc12') ;
%
%   There are several merge modes that can be selected using the
%   'mergeMode', option.
%
%   Let BT, BV, PT, PV, and PX be the Berkeley training and validation
%   sets and PASCAL segmentation challenge training, validation, and
%   test sets. Let T, V, X the final trainig, validation, and test
%   sets.
%
%   Mode 1::
%      V = PV (same validation set as PASCAL)
%
%   Mode 2:: (default))
%      V = PV \ BT (PASCAL val set that is not a Berkeley training
%      image)
%
%   Mode 3::
%      V = PV \ (BV + BT)
%
%   In all cases:
%
%      S = PT + PV + BT + BV
%      X = PX  (the test set is uncahgend)
%      T = (S \ V) \ X (the rest is training material)

opts.dataDir = 'data/voc12' ;
opts.archiveDir = 'data/archives' ;
opts.mergeMode = 2 ;
opts.url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz' ;
opts = vl_argparse(opts, varargin) ;

tempDir = fullfile(opts.dataDir, 'berkeley') ;
haveData = exist(tempDir);
if haveData
  % for real?
  files = dir(fullfile(opts.dataDir, 'berkeley', 'benchmark_RELEASE', 'dataset', 'cls', '*.mat')) ;
  haveData = length(files) > 0 ;
end

if ~haveData
  % Get Berkeley data
  archivePath = fullfile(opts.archiveDir, 'berkeleyVoc12Segments.tar.gz') ;
  if ~exist(archivePath)
    fprintf('%s: downloading %s to %s [this may take a long time]\n', mfilename, opts.url, archivePath) ;
    urlwrite(opts.url, archivePath) ;
  end

  % Uncompress Berkeley data
  mkdir(tempDir) ;
  untar(archivePath, tempDir) ;
end

mkdir(fullfile(opts.dataDir, 'SegmentationClassExt')) ;
mkdir(fullfile(opts.dataDir, 'SegmentationObjectExt')) ;

% Update training data
train = textread(fullfile(tempDir, 'benchmark_RELEASE', 'dataset', 'train.txt'), '%s','delimiter','\n') ;
val = textread(fullfile(tempDir, 'benchmark_RELEASE', 'dataset', 'val.txt'), '%s','delimiter','\n') ;

for i = 1:numel(imdb.images.id)
  name = imdb.images.name{i} ;
  isBT = any(find(strcmp(name, train))) ;
  isBV = any(find(strcmp(name, val))) ;

  isPT = imdb.images.segmentation(i) && imdb.images.set(i) == 1 ;
  isPV = imdb.images.segmentation(i) && imdb.images.set(i) == 2 ;
  isPX = imdb.images.segmentation(i) && imdb.images.set(i) == 3 ; % test

  % now decide how to use this image
  if ~(isBT || isBV || isPT || isPV || isPX)
    % not an image with segmentations
    continue ;
  end

  if isPX
    isX = true ;
    isT = false ;
    isV = false ;
  else
    switch opts.mergeMode
      case 1
        isV = isPV ;
      case 2
        isV = isPV & ~isBT ;
      case 3
        isV = isPV & ~isBT & ~isBV ;
    end
    isX = false ;
    isT = ~isV ;
  end

  % if this image is not in the berekeley data, copy it over
  % from the PASCAL DATA as is, otherwise use Berkely annotation
  for k = 1:2
    if k == 1
      dir1 = 'cls' ;
      dir2 = 'SegmentationClass' ;
      dir3 = 'SegmentationClassExt' ;
      f = 'GTcls' ;
    else
      dir1 = 'inst' ;
      dir2 = 'SegmentationObject' ;
      dir3 = 'SegmentationObjectExt' ;
      f = 'GTinst' ;
    end

    extPath = fullfile(tempDir, 'benchmark_RELEASE', 'dataset', dir1, [name '.mat']) ;
    pngPath = fullfile(opts.dataDir, dir2, [name '.png']) ;
    newPngPath = fullfile(opts.dataDir, dir3, [name '.png']) ;

    if ~exist(newPngPath)
      if imdb.images.segmentation(i)
        copyfile(pngPath, newPngPath, 'f') ;
      else
        anno = load(extPath) ;
        labels = anno.(f).Segmentation ;
        imwrite(uint8(labels),newPngPath) ;
      end
    end
  end

  imdb.images.segmentation(i) = true ;
  imdb.images.set(i) = isT + 2 * isV + 3 * isX ;
end


imdb.paths.classSegmentation = fullfile(opts.dataDir, 'SegmentationClassExt', '%s.png') ;
imdb.paths.objectSegmentation = fullfile(opts.dataDir, 'SegmentationObjectExt', '%s.png') ;

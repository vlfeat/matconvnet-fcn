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

opts.dataDir = 'data/voc12' ;
opts.archiveDir = 'data/archives' ;
opts.reduceValSet = true ;
opts.url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz' ;
opts = vl_argparse(opts, varargin) ;

% Get Berkeley data
archivePath = fullfile(opts.archiveDir, 'berkeleyVoc12Segments.tar.gz') ;
if ~exist(archivePath)
  fprintf('%s: downloading %s to %s\n', mfilename, opts.url, archivePath) ;
  urlwrite(opts.url, archivePath) ;
end

% Uncompress Berkeley data
tempDir = fullfile(opts.dataDir, 'berkeley') ;
if ~exist(tempDir)
  mkdir(tempDir) ;
  untar(archivePath, tempDir) ;
end

% Merge Berkeley data in PASCAL VOC format
for k = 1:2
  if k == 1
    dir1 = 'cls' ;
    dir2 = 'SegmentationClass' ;
    dir3 = 'SegmentationClassExt' ;
  else
    dir1 = 'cls' ;
    dir2 = 'SegmentationObject' ;
    dir3 = 'SegmentationObjectExt' ;
  end

  mkdir(fullfile(opts.dataDir, dir3)) ;

  for i = 1:numel(imdb.images.id)
    name = imdb.images.name{i} ;
    extPath = fullfile(tempDir, 'benchmark_RELEASE', 'dataset', dir1, [name '.mat']) ;
    pngPath = fullfile(opts.dataDir, dir2, [name '.png']) ;
    newPngPath = fullfile(opts.dataDir, dir3, [name '.png']) ;

    if exist(extPath)
      assert(imdb.images.set(i) < 3) ; % not test
      % found a Berkeley annotation
      % skip it if we want to use the original val set and this
      % is a validation image
      if (imdb.images.set(i) ~= 2 || opts.reduceValSet)
        anno = load(extPath) ;
        labels = anno.GTcls.Segmentation ;
        imwrite(uint8(labels),newPngPath) ;
        imdb.images.segmentation(i) = true ;
        imdb.images.set(i) = 1 ;
        continue ;
      end
    end
    if imdb.images.segmentation(i) & imdb.images.set(i) < 3
      copyfile(pngPath, newPngPath, 'f') ;
    end
  end
end

imdb.paths.classSegmentation = fullfile(opts.dataDir, 'SegmentationClassExt', '%s.png') ;
imdb.paths.objectSegmentation = fullfile(opts.dataDir, 'SegmentationObjectExt', '%s.png') ;

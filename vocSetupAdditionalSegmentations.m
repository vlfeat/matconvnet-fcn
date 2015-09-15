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
opts.preserveValSet = true ;
opts.url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz' ;
opts = vl_argparse(opts, varargin) ;

tempDir = fullfile(opts.dataDir, 'berkeley') ;
if ~exist(tempDir)
  % Get Berkeley data
  archivePath = fullfile(opts.archiveDir, 'berkeleyVoc12Segments.tar.gz') ;
  if ~exist(archivePath)
    fprintf('%s: downloading %s to %s\n', mfilename, opts.url, archivePath) ;
    urlwrite(opts.url, archivePath) ;
  end

  % Uncompress Berkeley data
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

  % Update training data
  train = textread(fullfile(tempDir, 'benchmark_RELEASE', 'dataset', 'train.txt'), '%s','delimiter','\n') ;
  val = textread(fullfile(tempDir, 'benchmark_RELEASE', 'dataset', 'val.txt'), '%s','delimiter','\n') ;

  for i = 1:numel(imdb.images.id)
    name = imdb.images.name{i} ;
    extPath = fullfile(tempDir, 'benchmark_RELEASE', 'dataset', dir1, [name '.mat']) ;
    pngPath = fullfile(opts.dataDir, dir2, [name '.png']) ;
    newPngPath = fullfile(opts.dataDir, dir3, [name '.png']) ;

    isTrain = any(find(strcmp(name, train))) ;
    isVal = any(find(strcmp(name, val))) ;

    % if this image is not in the berekeley data, copy it over
    if ~isTrain && ~isVal
      if imdb.images.segmentation(i) & imdb.images.set(i) < 3
        copyfile(pngPath, newPngPath, 'f') ;
      end
      continue ;
    end

    % skip if the image is a validation image in the berkeley data
    % and if we do not want to modify the validation set
    if isVal && opts.preserveValSet, continue ; end

    % skip if the image is a training image in the berkely data but
    % a val image in the PASCAL data and we do not want to modify
    % the validation set
    if isTrain && imdb.images.set(i) == 2, continue ; end

    assert(imdb.images.set(i) < 3) ; % not test

    anno = load(extPath) ;
    labels = anno.GTcls.Segmentation ;
    imwrite(uint8(labels),newPngPath) ;
    imdb.images.segmentation(i) = true ;
    imdb.images.set(i) = isTrain + 2 * isVal ;

  end
end

imdb.paths.classSegmentation = fullfile(opts.dataDir, 'SegmentationClassExt', '%s.png') ;
imdb.paths.objectSegmentation = fullfile(opts.dataDir, 'SegmentationObjectExt', '%s.png') ;

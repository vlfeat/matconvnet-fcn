function imdb = voc12_seg_extend(imdb, varargin)
% Download additional Berkeley segmentation data for VOC12

opts.dataDir = 'data/voc12' ;
opts.archiveDir = 'data/archives' ;
opts.url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz' ;

% Get Berkeley data
archivePath = fullfile(opts.archiveDir, 'berkelekyVoc12Segments.tar.gz') ;
if ~exist(archivePath)
  fprintf('%s: downloading %s to %s\n', mfilename, opts.url, archivePath) ;
  urlwrite(opts.url, archivePath) ;
end

% Uncompress Berkeley data
tempDir = fullfile(opts.dataDir, 'berkeley') ;
mkdir(tempDir) ;
untar(archivePath, tempDir) ;

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
    if imdb.images.set(i) == 1
      % for training images, use the Berkeley annotation if any
      extPath = fullfile(tempDir, 'dataset', dir1, [name '.mat']) ;
      pngPath = fullfile(opts.dataDir, dir2, [name '.png']) ;
      newPngPath = fullfile(opts.dataDir, dir3, [name '.png']) ;
      if exist(extPath)
        anno = load(extPath) ;
        imwrite(newPngPath, uint8(labels)) ;
        imdb.images.segmentation(i) = true ;
      elseif imdb.images.segmentation(i)
        % for val and test image, just use the original annotation
        copyfile(pngPath, newPngPath, 'f') ;
      end
    elseif imdb.images.segmentation(i)
      copyfile(pngPath, newPngPath, 'f') ;
    end
  end
end

imdb.paths.classSegmentation = fullfile(opts.dataDir, 'SegmentationClassExt', '%s.png') ;
imdb.paths.objectSegmentation = fullfile(opts.dataDir, 'SegmentationObjectExt', '%s.png') ;

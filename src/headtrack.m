function trajectory = headtrack(framesDir, varargin)
%HEADTRACK Track a head through an image sequence using its colour signature.
%
%   TRAJECTORY = HEADTRACK(FRAMESDIR) reads every PNG frame in FRAMESDIR
%   (sorted by name), asks the user to outline the head on the first frame,
%   and tracks it through the rest of the sequence. TRAJECTORY is an N-by-2
%   matrix with the [row col] centre of the tracked region in each frame.
%
%   The target is modelled by its chrominance (CbCr) histogram, the "colour
%   signature". In every new frame the signature is back-projected onto a
%   circular search zone and the tracking window is moved to the position
%   that maximises the concentration of signature pixels inside it.
%
%   HEADTRACK(..., 'Name', Value) accepts the following options:
%     'OutputVideo'  File name of a video with the tracking overlay
%                    (e.g. 'result.avi'). Default: '' (no video).
%     'FrameRate'    Frame rate of the output video. Default: 2.
%     'CropsDir'     Directory where the luminance of the tracked region
%                    is saved for every frame. Default: '' (do not save).
%     'Display'      Show every tracked frame on screen. Default: true.
%     'UpdateWeight' Weight of the new histogram when updating the
%                    signature, in [0 1]. Default: 0.5.
%
%   Example:
%     traj = headtrack('sequences/javi', 'OutputVideo', 'javi.avi');
%
%   See also TRACKROI, LOCATENEWROI, CBCRHISTOGRAM.
%
%   Javier Molina, Video Processing & Understanding Lab (VPU-Lab, formerly GTI), UAM, 2006.

p = inputParser;
p.addRequired('framesDir', @(s) ischar(s) || isstring(s));
p.addParameter('OutputVideo', '', @(s) ischar(s) || isstring(s));
p.addParameter('FrameRate', 2, @isnumeric);
p.addParameter('CropsDir', '', @(s) ischar(s) || isstring(s));
p.addParameter('Display', true, @(x) islogical(x) || isnumeric(x));
p.addParameter('UpdateWeight', 0.5, @(x) isnumeric(x) && x >= 0 && x <= 1);
p.parse(framesDir, varargin{:});
opt = p.Results;

frames = listFrames(char(framesDir));
if isempty(frames)
    error('headtrack:noFrames', 'No PNG frames found in "%s".', framesDir);
end
nFrames = numel(frames);

% --- Initialisation: the user outlines the head on the first frame ------
imageYCbCr = rgb2ycbcr(imread(frames{1}));

figure(1);
imagesc(ycbcr2rgb(imageYCbCr));
axis image;
title('Outline the head and double-click to finish');
roi = circularROI(roipoly);
close(1);

[~, signature] = cbcrHistogram(maskImage(imageYCbCr, roi));

trajectory = zeros(nFrames, 2);
trajectory(1, :) = roiCentre(roi);

if ~isempty(opt.CropsDir) && ~exist(opt.CropsDir, 'dir')
    mkdir(opt.CropsDir);
end
writer = [];
if ~isempty(opt.OutputVideo)
    writer = VideoWriter(char(opt.OutputVideo));
    writer.FrameRate = opt.FrameRate;
    open(writer);
end

showFrame(roi, imageYCbCr, 1, opt.Display, writer);

% --- Tracking loop -------------------------------------------------------
for k = 2:nFrames
    imageYCbCr = rgb2ycbcr(imread(frames{k}));

    [roi, histogram] = trackROI(roi, imageYCbCr, signature);

    % Blend the new observation into the colour signature so that it adapts
    % slowly to changes of illumination and pose.
    signature = (1 - opt.UpdateWeight) * signature + opt.UpdateWeight * histogram;

    trajectory(k, :) = roiCentre(roi);

    if ~isempty(opt.CropsDir)
        saveCrop(imageYCbCr(:, :, 1), roi, ...
            fullfile(opt.CropsDir, sprintf('crop%04d.png', k)));
    end

    showFrame(roi, imageYCbCr, k, opt.Display, writer);
end

if ~isempty(writer)
    close(writer);
end
end

% -------------------------------------------------------------------------
function frames = listFrames(framesDir)
% Sorted list of the PNG files in a directory (case-insensitive extension).
listing = dir(framesDir);
names = {listing(~[listing.isdir]).name};
names = sort(names(~cellfun(@isempty, regexpi(names, '\.png$'))));
frames = fullfile(framesDir, names);
end

function roi = circularROI(polygonMask)
% Replace the user-drawn polygon with the circle inscribed in its bounding
% box: heads are roughly round and a circle keeps the window symmetric.
[i, j] = find(polygonMask);
centre = round([(min(i) + max(i)) / 2, (min(j) + max(j)) / 2]);
radius = round(((max(i) - min(i)) + (max(j) - min(j))) / 4);

[cols, rows] = meshgrid(1:size(polygonMask, 2), 1:size(polygonMask, 1));
roi = double(hypot(rows - centre(1), cols - centre(2)) <= radius);
end

function masked = maskImage(imageYCbCr, roi)
masked = imageYCbCr .* uint8(repmat(roi, [1 1 3]));
end

function c = roiCentre(roi)
[i, j] = find(roi);
c = [mean(i), mean(j)];
end

function saveCrop(luma, roi, fileName)
[i, j] = find(roi);
imwrite(luma(min(i):max(i), min(j):max(j)), fileName);
end

function showFrame(roi, imageYCbCr, k, display, writer)
if ~display && isempty(writer)
    return;
end
overlay = addTrackingROI(edge(roi, 'canny'), imageYCbCr);
if display
    figure(2);
    imagesc(overlay);
    axis image;
    title(sprintf('Frame# %d', k));
    drawnow;
end
if ~isempty(writer)
    writeVideo(writer, overlay);
end
end

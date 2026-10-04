%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CS223b HW# 1b:
% Rohit Singh & Mitul Saha:  {rohitsi, mitul}@stanford.edu
%
% Lightly adapted for this repository: parameterised input/output and
% VideoWriter instead of the obsolete MakeQTMovie.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function lk(framePattern, nFrames, outputVideo)
%LK Lucas-Kanade tracker of a single user-selected region.
%
%   LK(FRAMEPATTERN, NFRAMES, OUTPUTVIDEO) reads grey-level frames named
%   sprintf(FRAMEPATTERN, k) for k = 1..NFRAMES, asks the user to select
%   the region to track on the first frame and writes a video with the
%   tracked region outlined in red.
%
%   Defaults: LK('1bw%d.png', 19, 'lk.avi').
%
%   Every frame pair is registered with an iterative (Newton-Raphson like)
%   Lucas-Kanade step: at most MAXITERATIONS iterations, stopping earlier
%   when the update in [u v] is below EPS pixels.

if nargin < 1, framePattern = '1bw%d.png'; end
if nargin < 2, nFrames = 19; end
if nargin < 3, outputVideo = 'lk.avi'; end

maxIterations = 15;    % maximum number of iterations per frame pair
eps           = 0.01;  % accuracy desired in terms of pixel width
sig           = 1.5;   % scale of the Gaussian derivative filters

writer = VideoWriter(outputVideo);
writer.FrameRate = 30;
open(writer);

F = double(imread(sprintf(framePattern, 1)));

% let the user select the region of interest
imagesc(F); colormap(gray); axis image;
roi = double(roipoly);

% edges of the ROI, to draw a nice tracking box
roiedges = double(edge(roi, 'canny'));

Gaus  = gauss(sig);
dGaus = dgauss(sig);
[rows, cols] = size(F);

for i = 2:nFrames
    fprintf('Iterating between frame %d and frame %d\n', i-1, i);

    G = double(imread(sprintf(framePattern, i)));
    Forig = F;
    Gorig = G;

    inds = find(roi);

    count = 0;
    u = 0; v = 0;  % F + [u v] == G
    du = 1; dv = 1;
    while ((abs(du) > eps || abs(dv) > eps) && count < maxIterations)
        count = count + 1;

        % smoothen and take derivative
        Fx = conv2(F, dGaus'*Gaus, 'same');
        Fy = conv2(F, Gaus'*dGaus, 'same');

        C = [sum(Fx(inds).^2)          sum(Fx(inds).*Fy(inds));
             sum(Fx(inds).*Fy(inds))   sum(Fy(inds).^2)];

        Ft = G - F;
        D  = [sum(Ft(inds).*Fx(inds)); sum(Ft(inds).*Fy(inds))];

        if rank(C) < 2, break; end

        uv = C \ D;
        du = uv(1); dv = uv(2);
        u = u + du;
        v = v + dv;

        % warp F
        [X1, Y1] = meshgrid((1:cols) + dv, (1:rows) + du);
        F = interp2(F, X1, Y1, 'cubic');
        F(isnan(F)) = 0;   % set the pixels warped in from outside to black
    end
    fprintf('No. of iterations: %d\n', count);

    if i == 2
        writeFrame(writer, overlay(roiedges, Forig), i-1);
    end

    % the shift can be fractional: round it to move the ROI
    ur = round(u); vr = round(v);
    offset = ur + vr*rows;

    roi      = shiftMask(roi, ur, vr, offset);
    roiedges = shiftMask(roiedges, ur, vr, offset);

    % if the window vanishes then we quit
    if ~any(roi(:))
        break;
    end

    writeFrame(writer, overlay(roiedges, G), i);
    F = Gorig;
end

close(writer);
close all;
end

function mask = shiftMask(mask, ur, vr, offset)
% Shift a mask by [-ur -vr], removing the parts outside the image.
[rows, cols] = size(mask);
[r, c] = find(mask > 0);
inds   = find(mask > 0);
inds(r-ur < 1 | r-ur > rows | c-vr < 1 | c-vr > cols) = [];
mask = mask * 0;
mask(inds - offset) = 1;
end

function colorimage = overlay(mask, im)
% Grey-level frame with the mask drawn in red.
red = (im - min(im(:))) * 0.5 / (max(im(:)) - min(im(:)));
green = red;
blue  = red;
red(mask > 0) = 1.0;
colorimage = cat(3, red, green, blue);
end

function writeFrame(writer, frame, k)
imagesc(frame); axis image;
title(sprintf('Frame# %d', k));
drawnow;
writeVideo(writer, getframe(gcf));
end

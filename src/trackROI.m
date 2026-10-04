function [newROI, newHist] = trackROI(roi, imageYCbCr, signature)
%TRACKROI Move the tracking window to its position in a new frame.
%
%   [NEWROI, NEWHIST] = TRACKROI(ROI, IMAGEYCBCR, SIGNATURE) locates the
%   binary window ROI in the frame IMAGEYCBCR using the normalised CbCr
%   histogram SIGNATURE of the target. It returns the displaced window and
%   the normalised CbCr histogram of the frame inside it, which the caller
%   uses to update the signature.
%
%   See also LOCATENEWROI, CBCRHISTOGRAM.

[i, j] = find(roi);

% The target is searched for within three times the window diagonal.
searchDiameter = 3 * floor(hypot(max(i) - min(i), max(j) - min(j)));

[dx, dy] = locateNewROI(roi, signature, imageYCbCr, searchDiameter);

newROI = shiftROI(roi, dx, dy);

[~, newHist] = cbcrHistogram(imageYCbCr .* uint8(repmat(newROI, [1 1 3])));
end

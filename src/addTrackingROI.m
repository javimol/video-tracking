function colorImage = addTrackingROI(mask, imageYCbCr)
%ADDTRACKINGROI Draw the outline of the tracking window over a frame.
%
%   COLORIMAGE = ADDTRACKINGROI(MASK, IMAGEYCBCR) converts the frame to RGB
%   and brightens the red and green channels on the pixels where MASK is
%   non-zero, so that the window outline shows up in yellow.

colorImage = ycbcr2rgb(imageYCbCr);
outline = repmat(mask > 0, [1 1 3]);
outline(:, :, 3) = false;

colorImage(outline) = colorImage(outline) + 100;   % uint8 saturates at 255
end

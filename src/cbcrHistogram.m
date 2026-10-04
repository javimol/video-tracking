function [hist, histNorm] = cbcrHistogram(imageYCbCr)
%CBCRHISTOGRAM Luminance-weighted chrominance histogram (colour signature).
%
%   [HIST, HISTNORM] = CBCRHISTOGRAM(IMAGEYCBCR) builds a 256-by-256
%   histogram indexed by (Cb, Cr) from a uint8 YCbCr image. Pixels outside
%   the region of interest must be set to zero: since RGB2YCBCR never
%   produces Cb = 0 or Cr = 0, those pixels are ignored.
%
%   Chrominance is unreliable in very dark or very bright pixels, so pixels
%   with a well-exposed luminance (60 < Y < 190) count 1 and the rest count
%   only 0.25.
%
%   HISTNORM is HIST linearly rescaled so that its peak is 255.

Y  = double(imageYCbCr(:, :, 1));
Cb = double(imageYCbCr(:, :, 2));
Cr = double(imageYCbCr(:, :, 3));

valid   = Cb > 0 & Cr > 0;
weights = 0.25 + 0.75 * (Y > 60 & Y < 190);

hist = accumarray([Cb(valid), Cr(valid)], weights(valid), [256 256]);

peak = max(hist(:));
if peak > 0
    histNorm = 255 * hist / peak;
else
    histNorm = hist;
end
end

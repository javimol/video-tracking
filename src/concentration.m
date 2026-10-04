function c = concentration(d, roi, likelihood, scale)
%CONCENTRATION Fraction of signature likelihood inside a displaced window.
%
%   C = CONCENTRATION(D, ROI, LIKELIHOOD, SCALE) shifts the binary window
%   ROI by round(D*SCALE) pixels and returns the mean of LIKELIHOOD (values
%   in [0 255]) over the shifted window, normalised to [0 1]. A value of 1
%   means every pixel in the window matches the colour signature perfectly.
%
%   See also LOCATENEWROI.

shift   = round(d * scale);
shifted = shiftROI(roi, shift(1), shift(2));

c = min(1, sum(likelihood(shifted > 0)) / (255 * nnz(roi)));
end

function [dx, dy, err] = locateNewROI(roi, signature, imageYCbCr, searchDiameter)
%LOCATENEWROI Find the displacement of the target in a new frame.
%
%   [DX, DY, ERR] = LOCATENEWROI(ROI, SIGNATURE, IMAGEYCBCR, SEARCHDIAMETER)
%   back-projects the normalised CbCr histogram SIGNATURE onto a circular
%   search zone of diameter SEARCHDIAMETER centred on the window ROI, and
%   returns the displacement [DX DY] (rows, columns) of the window that
%   maximises the concentration of signature pixels inside it.
%   ERR = 1 - concentration at the optimum.
%
%   See also CONCENTRATION, TRACKROI.

[rows, cols, ~] = size(imageYCbCr);

% Circular search zone around the current window.
[i, j] = find(roi);
centre = round([mean(i), mean(j)]);
[cc, rr] = meshgrid(1:cols, 1:rows);
searchZone = hypot(rr - centre(1), cc - centre(2)) <= searchDiameter / 2;

% Back-projection: every pixel takes the signature value of its (Cb, Cr).
Cb = double(imageYCbCr(:, :, 2));
Cr = double(imageYCbCr(:, :, 3));
valid = searchZone & Cb > 0 & Cr > 0;
likelihood = zeros(rows, cols);
likelihood(valid) = signature(sub2ind(size(signature), Cb(valid), Cr(valid)));

% Maximise the concentration with Nelder-Mead. fminsearch builds its
% initial simplex with a step of 0.00025 for zero-valued coordinates, so
% scaling the variables by 4000 makes that first step exactly one pixel.
scale = 4000;
[d, fval] = fminsearch(@(d) -concentration(d, roi, likelihood, scale), [0 0]);

dx  = round(d(1) * scale);
dy  = round(d(2) * scale);
err = 1 + fval;
end

function shifted = shiftROI(roi, dx, dy)
%SHIFTROI Displace a binary mask by DX rows and DY columns.
%
%   Pixels that fall outside the image are discarded.

[i, j] = find(roi);
i = i + dx;
j = j + dy;

inside = i >= 1 & i <= size(roi, 1) & j >= 1 & j <= size(roi, 2);

shifted = zeros(size(roi));
shifted(sub2ind(size(roi), i(inside), j(inside))) = 1;
end

%RUNCATWALK Batch run of the head tracker on the "pasarela" sequences.
%
%   The original experiments used five sequences recorded on the VPU-Lab
%   walkway ("pasarela"): people walking towards and away from the camera.
%   Each sequence is a folder of PNG frames. Set DATAROOT to the folder that
%   contains them; one AVI with the tracking overlay is written per sequence.

dataRoot  = fullfile('data', 'pasarela');
sequences = {'javi', 'javiEspalda', 'alvaro', 'alvaroEspalda', 'luis'};

for s = 1:numel(sequences)
    name = sequences{s};
    fprintf('Tracking sequence "%s"...\n', name);
    headtrack(fullfile(dataRoot, name), 'OutputVideo', [name '.avi']);
end

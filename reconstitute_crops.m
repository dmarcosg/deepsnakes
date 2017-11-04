crops_path = 'building_crops_gt';
result_path = 'result_binary';

try
    rmdir(result_path);
end

mkdir(result_path);

bb_names = dir(fullfile(crops_path,'*_bb.csv'));
imsize = [5000 5000];
for num = 1:numel(bb_names)
    im = zeros(imsize);
    imname = strsplit(bb_names(num).name,'_bb.csv');
    imname = imname{1};
    bb = csvread(fullfile(crops_path,bb_names(num).name));
    crop_names = dir(fullfile(crops_path,[imname,'*.png']));
    for i = 1:size(bb,1)
        crop = imread(fullfile(crops_path,crop_names(i).name));
        crop = imresize(crop,bb(i,[4 3]),'nearest');
        im(bb(i,2):bb(i,2)+bb(i,4)-1,bb(i,1):bb(i,1)+bb(i,3)-1) = crop + im(bb(i,2):bb(i,2)+bb(i,4)-1,bb(i,1):bb(i,1)+bb(i,3)-1);
    end
    im = min(im,1);
    imwrite(im,fullfile(result_path,[imname,'_snake.png']));
end
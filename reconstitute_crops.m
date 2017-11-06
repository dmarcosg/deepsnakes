intoronto = false;

crops_path = 'building_crops_gt';
result_path = 'result_binary';

if intoronto
    crops_path = '/ais/dgx1/marcosdi/TCityBuildings/building_crops';
    result_path = '/ais/dgx1/marcosdi/TCityBuildings/building_crops_gt';
else
    crops_path = '/home/diego/PycharmProjects/snakes_prj/deepsnakes/results/tcity1';
    result_path = '/home/diego/PycharmProjects/snakes_prj/deepsnakes/results/tcity1/val_tiles';
end

mkdir(result_path);

bb_names = dir(fullfile(crops_path,'*_bb.csv'));
imsize = [5000 5000];
for num = 1:numel(bb_names)
    im = uint16(zeros(imsize));
    imname = strsplit(bb_names(num).name,'_bb.csv');
    imname = imname{1};
    bb = csvread(fullfile(crops_path,bb_names(num).name));
    crop_names = dir(fullfile(crops_path,[imname,'*.png']));
    for i = 1:size(bb,1)
        crop = imread(fullfile(crops_path,crop_names(i).name));
        crop = crop(:,:,1);
        crop = imresize(crop,bb(i,[4 3]),'nearest');
        prev_crop = im(bb(i,2):bb(i,2)+bb(i,4)-1,bb(i,1):bb(i,1)+bb(i,3)-1);
        prev_crop(crop(:)>0) = i;
        im(bb(i,2):bb(i,2)+bb(i,4)-1,bb(i,1):bb(i,1)+bb(i,3)-1) = prev_crop;
    end
    imwrite(uint16(im),fullfile(result_path,[imname,'_snake.png']));
end

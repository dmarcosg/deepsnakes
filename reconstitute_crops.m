intoronto = false;

if intoronto
    crops_path = '/ais/dgx1/marcosdi/TCityBuildings/results1';
    result_path = '~/deepsnakes/results_val';
else
    crops_path = '/home/diego/PycharmProjects/snakes_prj/deepsnakes/results/tcity1';
    result_path = '/home/diego/PycharmProjects/snakes_prj/deepsnakes/results/tcity1/val_tiles';
end

mkdir(result_path);

bb_names = dir(fullfile(crops_path,'*_bb.csv'));
imsize = [5000 5000];
for num = 1:numel(bb_names)
    disp(['Doing tile ', bb_names(num).name]);
    im = uint16(zeros(imsize));
    imname = strsplit(bb_names(num).name,'_bb.csv');
    imname = imname{1};
    bb = csvread(fullfile(crops_path,bb_names(num).name));
    crop_names = dir(fullfile(crops_path,[imname,'*.png']));
    for i = 1:size(bb,1)
        crop = imread(fullfile(crops_path,crop_names(i).name));
        crop = crop(:,:,1);
        crop = imfill(crop,'holes');
        crop = imresize(crop,bb(i,[4 3]),'nearest');
        try
            prev_crop = im(bb(i,2):bb(i,2)+bb(i,4)-1,bb(i,1):bb(i,1)+bb(i,3)-1);
        catch
            disp('Bad bounding box');
            continue;
        end
        prev_crop(crop(:)>0) = i;
        im(bb(i,2):bb(i,2)+bb(i,4)-1,bb(i,1):bb(i,1)+bb(i,3)-1) = prev_crop;
    end
    imwrite(uint16(im),fullfile(result_path,[imname,'_snake.png']));
end

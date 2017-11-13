margin = 0.8;
s = 384;
minsize = 20; % minimum size of the dwt proposal, in pixels
training = false;
doplot = false;
do_dilate_init = false;
intoronto = true;

if do_dilate_init > 0
    strel = fspecial('disk',9);
    strel = strel / max(strel(:)) > 0.0;
end

if intoronto
    if training
        % Where to get the images from
        ims_path = '/ais/gobi4/justinliang/reconstruction/data/rotated_training_images/Images_RGB/';
        gt_path = '/ais/gobi4/justinliang/reconstruction/data/rotated_training_images/Labels/';
        %gt_path = '/ais/dgx1/marcosdi/improved_gt_train';
        dwt_path = '/ais/gobi4/TorontoCity/test/min/aerial_instances/joint_alignment_12_output/train/';
        % Where to store the cropped out buildings
        crops_path = '/ais/dgx1/marcosdi/TCityBuildings/building_crops';
            crops_gt_path = '/ais/dgx1/marcosdi/TCityBuildings/building_crops_gt';
        if do_dilate_init
            crops_dwt_path = '/ais/dgx1/marcosdi/TCityBuildings/building_crops_dwt_dilate';
        else
            crops_dwt_path = '/ais/dgx1/marcosdi/TCityBuildings/building_crops_dwt';
        end
    else
        % Where to get the images from
        ims_path = '/ais/gobi4/TorontoCity/data/Generated/CVPR_DemoArea/val/Images_RGB';
        dwt_path = '/ais/gobi4/TorontoCity/test/min/aerial_instances/joint_alignment_12_output/val';
        % Where to store the cropped out buildings
        crops_path = '/ais/dgx1/marcosdi/TCityBuildings/val_building_crops';
        if do_dilate_init
            crops_dwt_path = '/ais/dgx1/marcosdi/TCityBuildings/val_building_crops_dwt_dilate';
        else
            crops_dwt_path = '/ais/dgx1/marcosdi/TCityBuildings/val_building_crops_dwt';
        end
   end
else
    ims_path = '/mnt/bighd/Data/TorontoCityTile';
    gt_path = '/mnt/bighd/Data/TorontoCityTile';
    dwt_path = '/mnt/bighd/Data/TorontoCityTile';

    crops_path = '/mnt/bighd/Data/TorontoCityTile/building_crops';
    crops_gt_path = '/mnt/bighd/Data/TorontoCityTile/building_crops_gt';
    if do_dilate_init
        crops_dwt_path = '/mnt/bighd/Data/TorontoCityTile/building_crops_dwt_dilate';
    else
        crops_dwt_path = '/mnt/bighd/Data/TorontoCityTile/building_crops_dwt';
    end
end

if training
    ims = dir(fullfile(ims_path,'*_0.png'));
    %gts = dir(fullfile(gt_path,'*_instances.png'));
    gts = dir(fullfile(gt_path,'*_0_labels.png'));
    dwts = dir(fullfile(dwt_path,'*_binary.png'));
else
    ims = dir(fullfile(ims_path,'*.png'));
    dwts = dir(fullfile(dwt_path,'*_binary.png'));
end

try
    rmdir(crops_path);
    rmdir(crops_gt_path);
    rmdir(crops_dwt_path);
end

mkdir(crops_path);
mkdir(crops_dwt_path);
if training
    mkdir(crops_gt_path);
end


for num = 1:numel(ims)
    disp(['Image ',num2str(num),'/',num2str(numel(ims)),'...']);
    data_gt = [];
    data_dwt = [];
    bounding_boxes = [];
    count = 1;
    if training
        imname = strsplit(ims(num).name,'_0.png');
    else
        imname = strsplit(ims(num).name,'.png');
    end
    imname = imname{1};
    % read images and get instances
    try
        im = imread(fullfile(ims_path,ims(num).name));
    catch
        disp(['Couldnt read ',fullfile(ims_path,ims(num).name)]);
        continue;
    end
    try
        dwt = imread(fullfile(dwt_path,dwts(num).name));
    catch
        disp(['Couldnt read ',fullfile(dwt_path,dwts(num).name)]);
        continue;
    end
    if training
        try
            gt = imread(fullfile(gt_path,gts(num).name));
        catch
            disp(['Couldnt read ',fullfile(gt_path,gts(num).name)]);
            continue;
        end
        if size(gt,3) > 1
            building_map = gt(:,:,1)==255 & gt(:,:,2) == 0 & gt(:,:,3) == 0;
            building_map = gt(:,:,1) > 0;
            building_map = bwlabel(building_map);
        else
            building_map = gt;
        end
        %stats_gt = regionprops('table',building_map,building_map,'Centroid','BoundingBox','MeanIntensity');
        %assert(sum((stats_gt{1:end,'MeanIntensity'}' ~= (1:size(stats_gt,1))))==0);
    end
    dwt = bwlabel(dwt);
    stats = regionprops('table',dwt,dwt,'Centroid','BoundingBox','MeanIntensity');
    assert(sum((stats{1:end,'MeanIntensity'}' ~= (1:size(stats,1))))==0);
    
    % for each instance in the dwt
    for i = 1:size(stats,1)
        bb = stats{i,'BoundingBox'};
        m = mean(bb(3:4))*margin;
        % skip if too small object
        if m < minsize 
            continue;
        end
        % get crop around dwt instance
        bb = round(bb+[-m -m 2*m 2*m]);
        bb(3) = min(size(im,1),bb(1)+bb(3))-bb(1);
        bb(4) = min(size(im,2),bb(2)+bb(4))-bb(2);
        bb(1) = max(1,bb(1));
        bb(2) = max(1,bb(2));
        bb(3) = min(size(im,1),bb(3));
        bb(4) = min(size(im,2),bb(4));
        
        crop_dwt = imcrop(dwt,bb);
        
        
        if training
            % get combined dwt+gt crop
            crop_gt = imcrop(building_map,bb);
            gt_vals = crop_gt(crop_dwt(:)==stats{i,'MeanIntensity'});
            gt_vals = gt_vals(gt_vals>0);
            if isempty(gt_vals) || numel(gt_vals) < 100
                continue;
            end
            gt_ind = mode(gt_vals);
            crop_gt = imcrop(building_map,bb) == gt_ind;
            for er = 1:3
                crop_gt = imerode(crop_gt,[0 1 0; 1 1 1; 0 1 0]);
            end
            crop_gt = central_object(crop_gt);
            crop_gt = imresize(crop_gt,[s s],'nearest');
        end
        bounding_boxes(count,:) = bb;
        crop = imcrop(im,bb);
        crop_dwt = imcrop(dwt,bb) == i;
        if do_dilate_init
            crop_dwt = imdilate(crop_dwt,strel);
        end
        crop = imresize(crop,[s s]);
        crop_dwt = imresize(crop_dwt,[s s],'nearest');
        
        % get polygons
        if training
            B_gt = bwboundaries(crop_gt,'noholes');
            [ps_gt,ix_gt] = dpsimplify(B_gt{1},3);
            data_gt(count,1) = double(size(ps_gt,1));
            pst = ps_gt';
            if size(ps_gt,1) <= 3
                continue;
            end
            pst = pst(:);
            for j = 1:numel(pst)
                data_gt(count,1+j) = pst(j);
            end
        end
        
        B_dwt = bwboundaries(crop_dwt,'noholes');
        [ps_dwt,ix_dwt] = dpsimplify(B_dwt{1},6);
        data_dwt(count,1) = double(size(ps_dwt,1));
        pst = ps_dwt';
        if size(ps_dwt,1) <= 3
            continue;
        end
        pst = pst(:);
        for j = 1:numel(pst)
            data_dwt(count,1+j) = pst(j);
        end
        
        
        if doplot
            imagesc(crop);
            hold on
            imagesc(crop_dwt,'AlphaData',0.4);
            if training
                imagesc(crop_gt,'AlphaData',0.4);
                plot(ps_gt(:,2),ps_gt(:,1),'-o','LineWidth',3,'MarkerEdgeColor',[1,1,0]);
            end
            plot(ps_dwt(:,2),ps_dwt(:,1),'-o','LineWidth',3,'MarkerEdgeColor',[1,0,0]);
            hold off
            pause(1);
        end
        imwrite(crop,fullfile(crops_path,[imname,'_building_',num2str(count,'%0.4d'),'.png']));
        imwrite(crop_dwt,fullfile(crops_dwt_path,[imname,'_building_',num2str(count,'%0.4d'),'.png']));
        if training
            imwrite(crop_gt,fullfile(crops_gt_path,[imname,'_building_',num2str(count,'%0.4d'),'.png']));
        end
        count = count + 1;
    end
    disp(['Image ',num2str(num),'/',num2str(numel(ims)),' done.']);
    csvwrite(fullfile(crops_dwt_path,[imname,'_polygons.csv']),data_dwt);
    csvwrite(fullfile(crops_dwt_path,[imname,'_bb.csv']),bounding_boxes);
    csvwrite(fullfile(crops_path,[imname,'_bb.csv']),bounding_boxes);
    if training
        csvwrite(fullfile(crops_gt_path,[imname,'_polygons.csv']),data_gt);
        csvwrite(fullfile(crops_gt_path,[imname,'_bb.csv']),bounding_boxes);
    end
end

function bw = central_object(bw)
[X,Y] = meshgrid(1:size(bw,2),1:size(bw,1));
X = X - size(bw,2)/2;
Y = Y - size(bw,1)/2;
d = (X.^2 + Y.^2);
d = max(d(:)) - d;
[~,most_central_point] = max(d(:).*bw(:));
seed = false(size(bw));
seed(most_central_point) = true;
bw = imreconstruct(seed,bw);
end

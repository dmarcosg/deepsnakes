function extract_polygons(path)
class_label = [255];
files = dir(fullfile(path,'*.png'));
pattern = 'seg_\d\d\d.*';
do_save_mask = false;
do_save_class_mask = false;
data = [];

count = 1;
for num = 1:numel(files)
    match = regexp(files(num).name,pattern);
    if isempty(match) || match(1)~=1
        continue;
    else
        gt = imread(fullfile(path,files(num).name)); 
        buildings = true(size(gt,1),size(gt,2));
        for i = 1:numel(class_label)
            buildings = buildings & gt(:,:,i)==class_label(i);
        end
        the_building = central_object(buildings);
        B = bwboundaries(the_building);
        [ps,ix] = dpsimplify(B{1},3);
        subplot(3,3,mod(count,9)+1);
        imagesc(the_building);
        hold on
        plot(ps(:,2),ps(:,1),'-o','LineWidth',3,'MarkerEdgeColor',[1,1,0]);
        hold off
        xticks([])
        yticks([])
        pause(0.01);
        data(count,1) = double(size(ps,1));
        pst = ps';
        pst = pst(:);
        for i = 1:numel(pst)
            data(count,1+i) = pst(i);
        end
        if do_save_mask
            imwrite(the_building,fullfile(path,['building_mask_',num2str(count-1,'%0.3d'),'.png']));
        end
        if do_save_class_mask
            imwrite(buildings,fullfile(path,['all_buildings_mask_',num2str(count-1,'%0.3d'),'.png']));
        end
        count = count + 1;
    end
end
csvwrite(fullfile(path,'polygons.csv'),data);
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
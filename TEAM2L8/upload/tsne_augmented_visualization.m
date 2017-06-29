fname = '001001001.flex - Well A-1; Field #1.tif';
info = imfinfo(fname);
num_images = numel(info);
channel_1 = [];
channel_2 = [];
for k = 1:num_images
    data = imread(fname, k);
    if mod(k,2)==1
        channel_1 = cat(3, channel_1, data);
    end
    if mod(k,2)==0
        channel_2 = cat(3, channel_2, data);
    end
%     imagesc(data)
%     title(k);
%     pause
    % ... Do something with image A ...
end    

for k = 1:size(channel_1,3)
    subplot(1,2,1);
    imagesc(channel_1(:,:,k));
    subplot(1,2,2);
    imagesc(channel_2(:,:,k));
    title(k);
    pause
    % ... Do something with image A ...
end   


rough_forground_mask = (mean(channel_1,3)>60);
pixel_ind = find(rough_forground_mask==1);
data = reshape(channel_1, size(channel_1,1)*size(channel_1,2), size(channel_1,3));
data = data(pixel_ind,:);



subsampled_data = data(1:100:end,:);
subsampled_pixel_ind = pixel_ind(1:100:end);

mappedX = fast_tsne(subsampled_data);
plot(mappedX(:,1),mappedX(:,2),'.')

ns = createns(double(subsampled_data));
[idx, ~] = knnsearch(ns,double(data),'k',1);
upsampled_mappedX = mappedX(idx,:);


new_vis = zeros(size(channel_1,1), size(channel_1,2), 3);
new_vis(:,:,1) = mean(channel_1, 3);

tmp = zeros(size(channel_1,1), size(channel_1,2)); tmp(pixel_ind) = upsampled_mappedX(:,1);
new_vis(:,:,2) = tmp; 

tmp = zeros(size(channel_1,1), size(channel_1,2)); tmp(pixel_ind) = upsampled_mappedX(:,2);
new_vis(:,:,3) = tmp; 

new_vis2 = new_vis;
for k=1:3
    new_vis2(:,:,k) = ceil((new_vis(:,:,k) - min(min(new_vis(:,:,k)))) / (max(max(new_vis(:,:,k))) - min(min(new_vis(:,:,k)))) * 255);
end

save tsne_augmented_visualization
% 2022-07-03 Zhi Huang
close all; clear all; clc;

result_rootdir = 'results';
data_dir = 'datadir/';
cohort = 'HER2+';
%cohort = 'TNBC';

ratio = 32; % downsampling_ratio

switch cohort
    case 'HER2+'
        result_dir = [result_rootdir '/HER2+/'];
        HE_list = dir([data_dir 'HER2*(H&E).svs']);
        IHC_list = dir([data_dir 'HER2*(IHC).svs']);
    case 'TNBC'
        result_dir = [result_rootdir '/TNBC/'];
        HE_list = dir([data_dir 'TNBC*(H&E).svs']);
        IHC_list = dir([data_dir 'TNBC*(IHC).svs']);
end

gtruth = load([result_dir 'imageLabelingMask.mat']);
gt_source = gtruth.gTruth.DataSource.Source;
gt_label = gtruth.gTruth.LabelData;
gt_pid = {};
for i = 1:length(gt_source)
    pid = strsplit(gt_source{i}, '/');
    %pid = strsplit(pid{10}, '_');
    pid = pid{10};
    gt_pid{i} = pid;
end

fake_background_value = 243;
savepath = sprintf('%s3_tissues_%dx_downsampled/', result_dir, ratio);
if ~exist(savepath, 'dir');mkdir(savepath);end
if ~exist([savepath 'tissue'], 'dir');mkdir([savepath 'tissue']);end
if ~exist([savepath 'tissueMask'], 'dir');mkdir([savepath 'tissueMask']);end

savepath_imglbler_mask = sprintf('%s2_ImageLabeler_Polygon_%dx_downsampled/', result_dir, ratio);
if ~exist(savepath_imglbler_mask, 'dir');mkdir(savepath_imglbler_mask);end


%%%%%%%%%%%%%%%%%%


for i = 1:length(IHC_list)
    switch cohort
        case 'HER2+'
            id = IHC_list(i).name(6:8);
            id = strtrim(id); % remove leading and tailing space.
        case 'TNBC'
            id = IHC_list(i).name(6:8);
            id = strtrim(id); % remove leading and tailing space.
    end
    disp(id)
    
    he_downsampled = imread(sprintf('%s1_%dx_downsampled/%s_H&E.png',result_dir,ratio,id));
    ihc_downsampled = imread(sprintf('%s1_%dx_downsampled/%s_IHC.png',result_dir,ratio,id));
    
    Index_he = find(contains(gt_pid,[id '_H&E.png']));
    Index_ihc = find(contains(gt_pid,[id '_IHC.png']));

    tissue_contours_he = gt_label{Index_he,:};
    polygon_mat_he = [];
    for j = 1:length(tissue_contours_he)
        if length(tissue_contours_he{j}) > 0
            temp = tissue_contours_he{j};
            if length(temp) == 1
                temp = temp{1};
            end
            temp2 = [repmat(j,length(temp),1) temp];
            polygon_mat_he = [polygon_mat_he; temp2];
        end
    end
    csvwrite(sprintf('%s%s_H&E_polygon.csv',savepath_imglbler_mask,id), polygon_mat_he);


    tissue_contours_ihc = gt_label{Index_ihc,:};
    polygon_mat_ihc = [];
    for j = 1:length(tissue_contours_ihc)
        if length(tissue_contours_ihc{j}) > 0
            temp = tissue_contours_ihc{j};
            if length(temp) == 1
                temp = temp{1};
            end
            temp2 = [repmat(j,length(temp),1) temp];
            polygon_mat_ihc = [polygon_mat_ihc; temp2];
        end
    end
    csvwrite(sprintf('%s%s_IHC_polygon.csv',savepath_imglbler_mask,id), polygon_mat_ihc);
end

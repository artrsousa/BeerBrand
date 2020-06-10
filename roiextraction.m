clc;

files = dir('Dataset/');
files = files(3:size(files,1),1);
folders = files([files.isdir]);

counter = 0;
y = table();
for k = 1:length(folders)
    dirname = folders(k).name;
    rootname = 'Dataset/';
    
    imgs = ls(strcat(rootname,dirname,'/*.png'));
    imgs = natsortfiles(cellstr(imgs));
    
    y{counter+1:counter+length(imgs),1} = string(imgs);
    y{counter+1:counter+length(imgs),2} = string(dirname);
    w = warning ('off','all');
    
    for x = 1:length(imgs)
        Img = imread(strcat(rootname,dirname,'/',imgs{x}));
        fig = figure; imshow(Img);
        roi = drawrectangle;

        y{counter+x,3:6} = num2cell(roi.Position); 
        close(fig);
    end

    counter = counter + length(imgs);
end

writetable(y,'bbox.csv','QuoteStrings',false);
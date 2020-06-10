
function bboxes = bboxtable(train_file)
    bboxfile = readtable(train_file); %'train_matlab.csv'
    classes = unique(bboxfile.class);

    counter = 0;
    bboxes = table();
    for x = 1:length(classes)
        label = classes(x);
        selected = bboxfile(string(bboxfile.class)==label,:);

        for l = 1:size(selected,1)
            row = table2cell(selected(l,4:end));
            
            for cl = 1:size(row,2)
                if row{1,cl} < 0
                    row{1,cl} = row{1,cl}*-1;
                end
            end
            
            if row{1,1} + row{1,3} > 150
                row{1,1} = floor(row{1,1}) - 1;
                row{1,3} = floor(row{1,3}) - 1;
            end
            if row{1,2} + row{1,4} > 150
                row{1,2} = floor(row{1,2}) - 1;
                row{1,4} = floor(row{1,4}) - 1;
            end
            
            bboxes(counter+l,x) = {{cat(1,row{:})'}};
        end
        
        bboxes.Properties.VariableNames{x} = char(label);
        counter = counter + size(selected,1);
    end
    
    return;
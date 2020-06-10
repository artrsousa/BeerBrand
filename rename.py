import os

print("teste")

for dirname, dirs, filenames in os.walk('Dataset/'):
    for idx,file in enumerate(filenames):
        old = os.path.join(dirname,file)
        new = os.path.join(dirname,"img_"+str(idx)+".png")
        
        os.rename(old,new)
        
        
        # y = y.append(reader.iloc[counter:counter+len(filenames)])
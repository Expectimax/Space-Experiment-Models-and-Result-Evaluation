import os
# this file is used to include the correct class in the image name. The new name variable and the path need to be set
# accordingly
path = "C:/Users/ferdi/Intuitive_test/High"
new_name = "High"
filelist = os.listdir(path)


for file in filelist:
    print(file)
    old = os.path.join(path, file)
    print(old)
    new = f'{path}/{new_name}_{file}'
    print(new)
    os.rename(old, new)

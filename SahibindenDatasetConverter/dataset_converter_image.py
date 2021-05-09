import os
import glob
import shutil
import os
import termcolor

# ID-0   : Ankara
# ID-509 : Istanbul

_to = 'mydataset'
_from = 'dataset-istanbul'

os.makedirs(_to, exist_ok=True)

labelmap = {
    "1": "salon",
    "2": "mutfak",
    "3": "yatak",
    "4": "banyo",
}

ID = 509
for index, impath in enumerate(glob.glob(os.path.join(_from, '*/images'))):
    termcolor.cprint("Preparing for {}: {}".format(index, impath), 'red')

    for _, imgpath in enumerate(glob.glob(os.path.join(impath, '*'))):
        _, Id, _, Class =  imgpath.split(os.path.sep)
        Class, Type = Class.split(".")    
        target = os.path.join(_to, f"{ID+1}_{labelmap[Class]}.{Type}")
        shutil.copy(imgpath, target)
    termcolor.cprint("Preparing for {}".format(index), 'green')
    ID += 1

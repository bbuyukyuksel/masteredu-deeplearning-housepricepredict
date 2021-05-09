import glob
import json
import codecs
import os

_to = 'mydataset'
_from = 'dataset-istanbul'

annot_file = open(os.path.join(_to,'HousesInfo.txt'), 'w')
for _, jsonpath in enumerate(glob.glob(os.path.join(_from, '*/*.json'))):
    with codecs.open(jsonpath, 'r', 'utf-8') as jsonfile:
        price = json.load(jsonfile)["price"]
        annot_file.write(f"0 0 0 0 {price.replace('.', '')}\n")
annot_file.close()


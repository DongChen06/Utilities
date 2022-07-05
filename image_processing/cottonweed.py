import os
from os import walk
from shutil import copyfile


def main(source, dest):
    # generate output dataset
    if not os.path.exists(dest):
        os.makedirs(dest)

    index = 0
    for item in os.listdir(source):
        # get all the pictures in directory
        ext = (".JPEG", "jpeg", "JPG", ".jpg", ".png", "PNG")
        for (dirpath, dirnames, filenames) in walk(os.path.join(source, item)):
            for filename in filenames:
                if filename.endswith(ext):
                    image_name = item + '_' + str(index) + '.png'
                    copyfile(os.path.join(source, item, filename),
                             os.path.join(dest, image_name))
                    index += 1


if __name__ == "__main__":
    source = 'CottonWeedID15'
    dest = 'CottonWeedDiff'
    main(source, dest)

import os
import shutil

path = "Samples/"

subdirs = os.listdir(path)
for subdir in subdirs:
    files = os.listdir(path+subdir)
    n_files = len(files)

    if n_files <= 4:
        shutil.rmtree(path+subdir)

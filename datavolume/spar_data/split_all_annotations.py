import glob, random
from shutil import copyfile


random.seed(14)
# simple script
path_to_all_gold = "all_annotated/"
all_txt = glob.glob(path_to_all_gold + "*.txt")

random.shuffle(all_txt)

train_pairs = all_txt[:120]
test_pairs = all_txt[120:160]
val_pairs = all_txt[160:]


def copy_files_to_folders(list_of_texts, destination):
    for t in list_of_texts:
        gold_path, t_filename = t.rsplit('/', 1)
        a_filename = t_filename[:-3] + "ann"
        copyfile(t, destination + t_filename)
        copyfile(gold_path + '/' + a_filename, destination + a_filename)
    print("Done copying files for {}".format(destination))

copy_files_to_folders(train_pairs, "./train/")
copy_files_to_folders(val_pairs, "./val/")
copy_files_to_folders(test_pairs, "./test/")



import glob
import os

if __name__ == '__main__':
    dataset_folder_gmy="/home/ecust/txx/dataset/gas/IR/real/real_annotated_1"###########################################
    img_ext="png"########################################################################################################
    txt_folder="/home/ecust/txx/project/divmatch/DivMatch 1/datasets/real_anntated_1_voc/ImageSets/Main"################
    img_path_list=glob.iglob(os.path.join(dataset_folder_gmy,'*/image/*.{}'.format(img_ext)))
    train_list=[]
    val_list=[]
    test_list=[]
    for img_path in img_path_list:
        dataset_split=img_path.split('/')[-3]
        img_num=img_path.split('/')[-1].split('.')[0]
        if dataset_split in ["train"]:

            train_list.append(img_num)
        elif dataset_split in ["val"]:
            val_list.append(img_num)
        elif dataset_split in ["test"]:
            test_list.append(img_num)
        else:
            print("error!")
        txt_path = os.path.join(txt_folder, "{}.txt".format(dataset_split))
        with open(txt_path,"a") as f:
            f.write(img_num+"\n")
    print("len_train=",len(train_list))
    print("len_val=", len(val_list))
    print("len_test=", len(test_list))



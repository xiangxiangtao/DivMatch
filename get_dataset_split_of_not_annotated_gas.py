import glob
import os

if __name__ == '__main__':
    split_list=["test"]
    txt_folder="/home/ecust/txx/project/divmatch/DivMatch 1/datasets/real_annotated_2_voc/ImageSets/Main"
    for split in split_list:
        # img_folder="/home/ecust/txx/dataset/gas/IR/real/real_not_annotated/{}".format(split)###########################################
        img_folder="/home/ecust/txx/dataset/gas/IR/real/real_annotated_2/{}/image".format(split)
        img_list=os.listdir(img_folder)
        txt_path=os.path.join(txt_folder,'{}.txt'.format(split))
        for img in img_list:
            img_name=img[:img.index(".png")]
            with open(txt_path,"a") as f:
                f.write(img_name+"\n")
    print("end")



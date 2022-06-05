import os
import shutil

if __name__=="__main__":
    source_dataset_name="composite6"#################################
    cyclegan_output_folder="/home/ecust/txx/project/cycle_gan/pytorch-CycleGAN-and-pix2pix/results/gas_fake2real_cyclegan_CPRshift_2/fake2real/test_composite_6_train/images"###########add train and val
    target_folder="/home/ecust/txx/project/divmatch/DivMatch 1/datasets/{}CPR/JPEGImages".format(source_dataset_name)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    image_list=os.listdir(cyclegan_output_folder)
    for image in image_list:
        # print(image)
        cat=image[image.index("_")+1:image.index(".png")]
        if cat in ["fake"]:
            print(image)
            new_name=image[:image.index("_")]+".jpg"
            shutil.copyfile(os.path.join(cyclegan_output_folder,image),os.path.join(target_folder,new_name))
    print("end")
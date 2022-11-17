from email.mime import image
import os
import shutil
import random
from datetime import datetime
import cv2
random.seed(datetime.now())


file = os.listdir("./data/thesis_left_20220710/")
# for img_name file:
    # img = cv2.imread("/media/cihci/0000678400004823/andy_master/code/ACVNet/data/train_disparity/"+str(img_name))
    # img.shape
    # print(img.shape)
    # img_name1 = str(img_name)
    # img_name1 = img_name1.split("_",1)
    # img_name1 = img_name1[1]
    # source_disparity = str("/media/cihci/0000678400004823/andy_master/code/Mask_RCNN/result/disparity_max192_all_select/"+img_name)
    # destination_disparity =str("/media/cihci/0000678400004823/andy_master/code/ACVNet/data/train_disparity_192/"+img_name) 
    # shutil.copyfile(source_disparity, destination_disparity)

for img_name in file:
    print(str(file.index(img_name))+"/"+str(len(file)))
    img_name1 = str(img_name)
    img_name1 = img_name1.split(".",1)
    img_name1 = img_name1[0]
    # img = cv2.imread(str("/media/cihci/0000678400004823/andy_master/code/Mask_RCNN/data/left/"+img_name1+".jpg"))
    # img = cv2.resize( img, (1241,376), interpolation=cv2.INTER_AREA)
    # cv2.imwrite( str("/media/cihci/0000678400004823/andy_master/code/ACVNet/data/train_left_20220707/"+img_name1+".png"),img)
    # img = cv2.imread(str("/media/cihci/0000678400004823/andy_master/code/Mask_RCNN/data/right/"+img_name1+".jpg"))
    # img = cv2.resize( img, (1241,376), interpolation=cv2.INTER_AREA)
    # cv2.imwrite( str("/media/cihci/0000678400004823/andy_master/code/ACVNet/data/train_right_20220707/"+img_name1+".png"),img)
    img = cv2.imread(str("./data/thesis_left_20220710/"+img_name))
    img = cv2.resize( img, (1241,376), interpolation=cv2.INTER_AREA)
    cv2.imwrite( str("./data/thesis_left_20220710/"+img_name1+".png"),img)
    img = cv2.imread(str("./data/thesis_right_20220710/"+img_name))
    img = cv2.resize( img, (1241,376), interpolation=cv2.INTER_AREA)
    cv2.imwrite( str("./data/thesis_right_20220710/"+img_name1+".png"),img)

    path = "./filenames/test_20220710.txt"
    f = open(path, 'a')
    f.write(str("data/thesis_left_20220710/"+img_name1 +".png data/thesis_right_20220710/"+img_name1+".png"))
    f.write("\n")
    f.close()
'''
    source = str("/media/cihci/0000678400004823/andy_master/code/Mask_RCNN/data/left/"+img_name1+".jpg")
    destination = str("/media/cihci/0000678400004823/andy_master/code/ACVNet/data/train_left_192/"+img_name1+".jpg")
    shutil.copyfile(source, destination)


    source = str("/media/cihci/0000678400004823/andy_master/code/Mask_RCNN/data/right/"+img_name1+".jpg")
    destination = str("/media/cihci/0000678400004823/andy_master/code/ACVNet/data/train_right_192/"+img_name1+".jpg")
    shutil.copyfile(source, destination)

    source = str("/media/cihci/0000678400004823/andy_master/code/Mask_RCNN/result/disparity_max192_all_select/"+img_name)
    destination = str("/media/cihci/0000678400004823/andy_master/code/ACVNet/data/train_disparity_192/"+img_name1+".jpg")
    shutil.copyfile(source, destination)
'''
# for img_name in file:
#     file.remove(img_name)
#     # train_file = file
#     # img_name1 = str(img_name)
#     # img_name1 = img_name1.split(".",1)
#     # img_name = img_name1[0]
#     path = './filenames/testing_all_20220707.txt'
#     f = open(path, 'a')
#     f.write(str("data/train_left_20220707/"+img_name +" data/train_right_20220707/"+img_name))
#     f.write("\n")
#     f.close()

# test_file = random.sample(file, 2649)
# print(len(test_file))
# for img_name in test_file:
#     file.remove(img_name)
#     train_file = file
#     # img_name1 = str(img_name)
#     # img_name1 = img_name1.split(".",1)
#     # img_name = img_name1[0]
#     path = './filenames/test_192_20220707.txt'
#     f = open(path, 'a')
#     f.write(str("data/train_left_20220707/"+img_name +" data/train_right_20220707/"+img_name+" "+"data/train_disparity_20220707/"+img_name))
#     f.write("\n")
#     f.close()



# for img_name in train_file:
#     # img_name1 = str(img_name)
#     # img_name1 = img_name1.split(".",1)
#     # img_name = img_name1[0]
#     path = './filenames/train_192_20220707.txt'
#     f = open(path, 'a')
#     f.write(str("data/train_left_20220707/"+img_name+" "+"data/train_right_20220707/"+img_name+" "+"data/train_disparity_20220707/"+img_name))
#     f.write("\n")
#     f.close()


# for i in test_file:
#     for j in train_file:
#         if i==j:
#             print("error")


# for img_name in file:
#     img_name1 = str(img_name)
#     img_name1 = img_name1.split("_",1)
#     img_name1 = img_name1[1]
#     path = './data/train.txt'
#     # f = open(path, 'a')
#     # f.write(str("data/train_left/"+img_name1+" "+"data/train_right/"+img_name1+" "+"data/train_disparity/"+img_name))
#     # f.write("\n")
#     source_left = str("/media/cihci/My Passport/data/left/"+img_name1)
#     destination_left =str("/media/cihci/0000678400004823/andy_master/code/ACVNet/data/train_left/"+img_name1) 
#     shutil.copyfile(source_left, destination_left)


#     source_right = str("/media/cihci/My Passport/data/right/"+img_name1)
#     destination_right =str("/media/cihci/0000678400004823/andy_master/code/ACVNet/data/train_right/"+img_name1) 
#     shutil.copyfile(source_right, destination_right)

#     source_disparity = str("/media/cihci/0000678400004823/andy_master/code/ACVNet/data/train_disparity/"+img_name)
#     destination_disparity =str("/media/cihci/0000678400004823/andy_master/code/ACVNet/data/train_disparity_only_filename/"+img_name1) 
#     shutil.copyfile(source_disparity, destination_disparity)

    # f.close()

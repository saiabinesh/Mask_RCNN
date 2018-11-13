
import os

ROOT_DIR = os.path.abspath("../")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
print("IMAGE_DIR")
print(IMAGE_DIR,"\n")

path_to_watch = [os.path.join(os.path.dirname(os.path.abspath(__file__)),IMAGE_DIR,i) for i in os.listdir(IMAGE_DIR)]
print(path_to_watch)


# print(os.path.dirname(IMAGE_DIR))
# # list_of_files = glob.glob('C:\Projects\Mask_RCNN\old_images\scale 5 - 4 places*') # * means all if need specific format then *.csv
# # print(list_of_files)
# print("list_dir")
# #list_dir = os.listdir('C:\Projects\Mask_RCNN\old_images\scale 5 - 4 places')
# dir='C:\Projects\Mask_RCNN\old_images\scale 5 - 4 places'
# list_dir = [os.path.join(os.path.dirname(os.path.abspath(__file__)),dir,i) for i in os.listdir(dir)]
# print(list_dir)
# # latest_file = max(list_of_files, key=os.path.getctime)
# # print (latest_file)
# latest_file_dir = max(list_dir , key=os.path.getctime)
# print (latest_file_dir)

IMAGE_DIR_formatted = [os.path.join(os.path.dirname(os.path.abspath(__file__)),IMAGE_DIR,i) for i in os.listdir(IMAGE_DIR)] #path to find latest file has to be in a different format with \\ rather than \ in DEFAULT IMAGE_DIR

image_name = max(IMAGE_DIR_formatted , key=os.path.getctime) #taking the latest file in for detection 
print(image_name)
print(os.path.basename(image_name))
# skipped configdisplay() for debugging

global categories_list


import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import skimage.color
import skimage.io
import skimage.transform

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
print(ROOT_DIR)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

#%matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    print("coco model path not found. Downloading")
    utils.download_trained_weights(COCO_MODEL_PATH)

## Configurations

#Need to replace the following hard caoded category list with the appropriate attribute of coco class
categories_list =  [{"supercategory": "person", "id": 1, "name": "person"}, {"supercategory": "vehicle", "id": 2, "name": "bicycle"}, {"supercategory": "vehicle", "id": 3, "name": "car"}, {"supercategory": "vehicle", "id": 4, "name": "motorcycle"}, {"supercategory": "vehicle", "id": 5, "name": "airplane"}, {"supercategory": "vehicle", "id": 6, "name": "bus"}, {"supercategory": "vehicle", "id": 7, "name": "train"}, {"supercategory": "vehicle", "id": 8, "name": "truck"}, {"supercategory": "vehicle", "id": 9, "name": "boat"}, {"supercategory": "outdoor", "id": 10, "name": "gas cylinder"}, {"supercategory": "outdoor", "id": 11, "name": "fire hydrant"}, {"supercategory": "outdoor", "id": 13, "name": "stop sign"}, {"supercategory": "outdoor", "id": 14, "name": "parking meter"}, {"supercategory": "outdoor", "id": 15, "name": "bench"}, {"supercategory": "animal", "id": 16, "name": "bird"}, {"supercategory": "animal", "id": 17, "name": "cat"}, {"supercategory": "animal", "id": 18, "name": "dog"}, {"supercategory": "animal", "id": 19, "name": "horse"}, {"supercategory": "animal", "id": 20, "name": "sheep"}, {"supercategory": "animal", "id": 21, "name": "cow"}, {"supercategory": "animal", "id": 22, "name": "elephant"}, {"supercategory": "animal", "id": 23, "name": "bear"}, {"supercategory": "animal", "id": 24, "name": "zebra"}, {"supercategory": "animal", "id": 25, "name": "giraffe"}, {"supercategory": "accessory", "id": 27, "name": "backpack"}, {"supercategory": "accessory", "id": 28, "name": "umbrella"}, {"supercategory": "accessory", "id": 31, "name": "handbag"}, {"supercategory": "accessory", "id": 32, "name": "tie"}, {"supercategory": "accessory", "id": 33, "name": "suitcase"}, {"supercategory": "sports", "id": 34, "name": "frisbee"}, {"supercategory": "sports", "id": 35, "name": "skis"}, {"supercategory": "sports", "id": 36, "name": "snowboard"}, {"supercategory": "sports", "id": 37, "name": "sports ball"}, {"supercategory": "sports", "id": 38, "name": "kite"}, {"supercategory": "sports", "id": 39, "name": "baseball bat"}, {"supercategory": "sports", "id": 40, "name": "baseball glove"}, {"supercategory": "sports", "id": 41, "name": "skateboard"}, {"supercategory": "sports", "id": 42, "name": "surfboard"}, {"supercategory": "sports", "id": 43, "name": "tennis racket"}, {"supercategory": "kitchen", "id": 44, "name": "bottle"}, {"supercategory": "kitchen", "id": 46, "name": "wine glass"}, {"supercategory": "kitchen", "id": 47, "name": "cup"}, {"supercategory": "kitchen", "id": 48, "name": "fork"}, {"supercategory": "kitchen", "id": 49, "name": "knife"}, {"supercategory": "kitchen", "id": 50, "name": "spoon"}, {"supercategory": "kitchen", "id": 51, "name": "bowl"}, {"supercategory": "food", "id": 52, "name": "banana"}, {"supercategory": "food", "id": 53, "name": "apple"}, {"supercategory": "food", "id": 54, "name": "sandwich"}, {"supercategory": "food", "id": 55, "name": "orange"}, {"supercategory": "food", "id": 56, "name": "broccoli"}, {"supercategory": "food", "id": 57, "name": "carrot"}, {"supercategory": "food", "id": 58, "name": "hot dog"}, {"supercategory": "food", "id": 59, "name": "pizza"}, {"supercategory": "food", "id": 60, "name": "donut"}, {"supercategory": "food", "id": 61, "name": "cake"}, {"supercategory": "furniture", "id": 62, "name": "chair"}, {"supercategory": "furniture", "id": 63, "name": "couch"}, {"supercategory": "furniture", "id": 64, "name": "potted plant"}, {"supercategory": "furniture", "id": 65, "name": "bed"}, {"supercategory": "furniture", "id": 67, "name": "dining table"}, {"supercategory": "furniture", "id": 70, "name": "toilet"}, {"supercategory": "electronic", "id": 72, "name": "tv"}, {"supercategory": "electronic", "id": 73, "name": "laptop"}, {"supercategory": "electronic", "id": 74, "name": "mouse"}, {"supercategory": "electronic", "id": 75, "name": "remote"}, {"supercategory": "electronic", "id": 76, "name": "keyboard"}, {"supercategory": "electronic", "id": 77, "name": "cell phone"}, {"supercategory": "appliance", "id": 78, "name": "microwave"}, {"supercategory": "appliance", "id": 79, "name": "oven"}, {"supercategory": "appliance", "id": 80, "name": "toaster"}, {"supercategory": "appliance", "id": 81, "name": "sink"}, {"supercategory": "appliance", "id": 82, "name": "refrigerator"}, {"supercategory": "indoor", "id": 84, "name": "book"}, {"supercategory": "indoor", "id": 85, "name": "clock"}, {"supercategory": "indoor", "id": 86, "name": "vase"}, {"supercategory": "indoor", "id": 87, "name": "scissors"}, {"supercategory": "indoor", "id": 88, "name": "teddy bear"}, {"supercategory": "indoor", "id": 89, "name": "hair drier"}, {"supercategory": "indoor", "id": 90, "name": "toothbrush"}]

class aerial_trains_Config(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "aerial_trains"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 2048

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (64, 128, 256, 512, 1024)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 10

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 1
    
config = aerial_trains_Config()
# config.display()

# ## Notebook Preferences

# def get_ax(rows=1, cols=1, size=8):
    # """Return a Matplotlib Axes array to be used in
    # all visualizations in the notebook. Provide a
    # central point to control graph sizes.
    
    # Change the default size attribute to control the size
    # of rendered images
    # """
    # _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    # return ax



import cv2
class aerial_trains_Dataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    def load_coco(self, dataset_dir, subset, year="2014", class_ids=None,
              class_map=None, return_coco=False, auto_download=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        if auto_download is True:
            self.auto_download(dataset_dir, subset, year)

        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco
    
    #Using the default from the base class (utils.py) and pasting here for clarity 

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        print("loading image: ",image_id)
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image
    
    def load_mask(self, image_id): #image_id here is just the serial number of the images in image_info, not their unique id in the json. 
        """Generate instance masks for shapes of the given image ID.
        """
        global categories_list
        print("loading mask for image_id: ",image_id)
        # Replacing self with dataset_train
        print("image_info  = ", dataset_train.image_info[image_id])
        ##############doing this only for train. Need to change 
        # extract all the unique objects from the annotation and create counters for them
        list_of_object_cat_ids = list(set([dict['category_id'] for dict in dataset_train.image_info[image_id]["annotations"]])) #this returns a unique list of all category_ids in the annotation 
        number_of_objects_in_image = len(list_of_object_cat_ids)
		#Extracting the corresponding object names based on the category ids
        list_object_names = [categories_list[i-1]["name"] for i in list_of_object_cat_ids]
        #creating a list of 1s that can be used as object counters for images with multiple objects
        # object_counter_dict = dict((key, value) for (key, value) in zip(list_object_names, [1]*number_of_objects_in_image))
        # for each unique object, create a counters. Increment them when one of them is being loaded. and use the counter number to extract the mask using the filename
		# Create counters for different objects 
        mask = np.ones(shape = (2000,2000,len(dataset_train.image_info[image_id]["annotations"])))
        mask_counter = 0 
		#The actual unique image_id written in the json which can be used to call the correspoding original raw images of the format raw_image_1.png , where 1 is the image_id (image_number as called here)
        image_number = dataset_train.image_info[image_id]["id"]
        print("Doing image with image_id: ", dataset_train.image_info[image_id]["id"])
        temp_class_id = []
        for annotation in dataset_train.image_info[image_id]["annotations"]:
            print("Current Annotation processed :", annotation)
            current_category_id = annotation["category_id"]
            temp_class_id.append(current_category_id)
            # print("current_category_id = ", current_category_id)         
            current_object_name = categories_list[current_category_id-1]["name"]
            # print("current_object_name: ",current_object_name)
            # current_object_instance_number = object_counter_dict[current_object_name]
            current_object_instance_number = annotation["instance_number"]
           # print("current_object_instance_number", current_object_instance_number)
            # object_counter_dict[current_object_name] = object_counter_dict[current_object_name]+1
            #access the mask from filename of the object mask for that image	
            # ex D:\AirSim\New\Images\Images_master
            
            # changed to double underscore here to check if filename path throws error.
            color_mask_filename = "D:/AirSim/New/Images/Images_master/image_"+str(image_number)+"_"+str(current_object_name)+"_"+str(current_object_instance_number)+".png" 	
            if os.path.isfile(color_mask_filename):
                img = cv2.imread(color_mask_filename,0)   
                ret,current_mask  = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
                #print("Type of object: ",type(mask))
                mask[:,:,mask_counter] = current_mask
                mask_counter = mask_counter + 1
                #extracting class id from the coco dictionary of the json annotations
            else:
                print("Mask file for image_number "+str(image_number)+" is missing from path: "+str(color_mask_filename))
        print("Temp class id from list unpacking: ",temp_class_id)               
        #Commenting out all the old code for class ids to extract it from the list
        class_id = [dict['category_id'] for dict in dataset_train.image_info[image_id]["annotations"]]
        print(class_id)
        #class_id = self.image_info[image_id]["annotations"][0]["category_id"]
        class_id = np.asarray(class_id)
        # print(np.shape(class_id))
        # class_id.shape = (1,)
        #print("class id: ",class_id)
        #print("class id shape ",np.shape(class_id))
        # _idx = np.sum(mask, axis=(0, 1)) > 0
        #print("_idx value = ",_idx)
        # mask = mask[:, :, _idx]

        return mask.astype(np.bool), class_id

    def random_shape(self, height, width):
        """Generates specifications of a random shape that lies within
        the given height and width boundaries.
        Returns a tuple of three valus:
        * The shape name (square, circle, ...)
        * Shape color: a tuple of 3 values, RGB.
        * Shape dimensions: A tuple of values that define the shape size
                            and location. Differs per shape type.
        """
        # Shape
        shape = random.choice(["square", "circle", "triangle"])
        # Color
        color = tuple([random.randint(0, 255) for _ in range(3)])
        # Center x, y
        buffer = 20
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        # Size
        s = random.randint(buffer, height//4)
        return shape, color, (x, y, s)

    def random_image(self, height, width):
        """Creates random specifications of an image with multiple shapes.
        Returns the background color of the image and a list of shape
        specifications that can be used to draw the image.
        """
        # Pick random background color
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        # Generate a few random shapes and record their
        # bounding boxes
        shapes = []
        boxes = []
        N = random.randint(1, 4)
        for _ in range(N):
            shape, color, dims = self.random_shape(height, width)
            shapes.append((shape, color, dims))
            x, y, s = dims
            boxes.append([y-s, x-s, y+s, x+s])
        # Apply non-max suppression wit 0.3 threshold to avoid
        # shapes covering each other
        keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(N), 0.3)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return bg_color, shapes

###########################Change directory for dataset here#####################
dataset_dir = "D:/AirSim/New/Images/coco"
# Training dataset
dataset_train = aerial_trains_Dataset()
dataset_train.load_coco(dataset_dir, "train", year = "2014",return_coco=True)
#dataset_train.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

#print(dataset_train.image_info)

# Validation dataset
dataset_val = aerial_trains_Dataset()
dataset_val.load_coco(dataset_dir, "val", year = "2014",return_coco=True)
#dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()
# print(dataset_train.image_info[0:10])
# exit()

### Exploring the structure of image_info to get class_ids

#print(dataset_train.image_info[0]["annotations"][0]["category_id"])
######Testing code for load_image_gt in model.py which is throwing error
#print(range(1,len(dataset_train.image_info)))
#print(dataset_train.image_info)
#print(dataset_train.image_info[0]["id"])
#k=2
#[dataset_train.image_info[x]["annotations"][0]["category_id"]
 #for x in range(0,len(dataset_train.image_info)) if dataset_train.image_info[x]["id"]==k]
#mask, class_ids_calculated = dataset_train.load_mask(2)

#List comprehension to load the path of the image corresponding to the image_id for the load_image function
# image_id = 0
# #[dataset_train.image_info[x]["path"]
 # #for x in range(0,len(dataset_train.image_info)) if dataset_train.image_info[x]["id"]==image_id]
# image_number = dataset_train.image_info[image_id]["id"]
# print(image_number)
# print(dataset_train.image_info[image_id]["annotations"][0]["category_id"])



#Load and display images of train with just single trains and masks
image_ids = np.asarray([91,11,42])
print("image_ids: ",image_ids)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


	

# ########################################## Training phase #######################################	
# # Create model in training mode
# model = modellib.MaskRCNN(mode="training", config=config,
                          # model_dir=MODEL_DIR)

# # Which weights to start with?
# init_with = "coco"  # imagenet, coco, or last

# if init_with == "imagenet":
    # model.load_weights(model.get_imagenet_weights(), by_name=True)
# elif init_with == "coco":
    # # Load weights trained on MS COCO, but skip layers that
    # # are different due to the different number of classes
    # # See README for instructions to download the COCO weights
    # model.load_weights(COCO_MODEL_PATH, by_name=True,
                       # exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                # "mrcnn_bbox", "mrcnn_mask"])
# elif init_with == "last":
    # # Load the last model you trained and continue training
    # model.load_weights(model.find_last()[1], by_name=True)


# # Train the head branches
# # Passing layers="heads" freezes all layers except the head
# # layers. You can also pass a regular expression to select
# # which layers to train by name pattern.
# model.train(dataset_train, dataset_val, 
            # learning_rate=config.LEARNING_RATE, 
            # epochs=10, 
            # layers='heads')
			
			
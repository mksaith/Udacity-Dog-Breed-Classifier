from IPython.display import display, HTML
from IPython.display import display_html
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2      
import matplotlib.pyplot as plt 
from torchvision import datasets,transforms
import time
import models as mds
import random
import math
import torchvision.transforms.functional as F
import numpy as np

def get_name_from_filename(file_name):
    name = file_name.split('/')
    name = name[len(name)-1].split('.')
    name = name[len(name)-2].split('_')
    name = ' '.join(name[:len(name)-1])
    
    return name

def show_images(image_files, n_col, image_size, h, w, titles = None):
    # following if condition is to avoid error:
   #         ValueError: Sample larger than population or is negative
    n_images = len(image_files)
    row=math.ceil(n_images/n_col);col=n_col
    
    
    n_file=0
    for rw in range(row):
        fig = plt.figure(figsize=(h, w))
        for cl in range(col):
            # load color (BGR) image
            image_file = image_files[n_file]
            image = cv2.imread(image_file)

            # convert BGR image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # extract pre-trained face detector
            face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

            # find faces in image
            faces = face_cascade.detectMultiScale(gray)

            # get bounding box for each detected face
            for (x,y,w,h) in faces:
                # add bounding box to color image
                cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

            image = cv2.resize(image,(100,100))

            # convert BGR image to RGB for plotting
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            fig.add_subplot(1, col, cl+1)
            
            if titles == None:
                title = get_name_from_filename(image_file)
            else:
                title = titles[n_file]
                
            plt.title(title)
                
            plt.imshow(image_rgb)
            
            
            n_file += 1
            if n_file+1>len(image_files):
                return
        plt.tight_layout()
        plt.show();
            
def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]



def ShowImgMap(img_map, m, h, w, total_time):
    """
    show 10 convoluted images i.e. the first 5 and the last 5
    """
    start = time.time()
    
    img_map = m(img_map)
    
    end=time.time()-start
    total_time += end
    
    fig=plt.figure(figsize=(h, w))

    imgs = img_map.squeeze(0)
    print('-'*127)
    
    print('Module: {}'.format(m))
    print('Output size( {}x{}x{} ): {}     Time: {}'.format(img_map.shape[1], img_map.shape[2], img_map.shape[3], \
                                                            img_map.shape[1]*img_map.shape[2]*img_map.shape[3], end))
    print('-'*127)
    channels = len(imgs)
    
    row=1
    col=10
    for i, idx in enumerate(range(0, min(channels, 10))): #, 3, 4, channels-6, channels-5 ,channels-4,channels-3,channels-2]):
        fig.add_subplot(row, col, i+1)
        plt.imshow(imgs[idx,:,:].detach().numpy())
    plt.tight_layout()    
    plt.show()
    
    return img_map, total_time
    
def visualize_model_cnn_output(image_size, model, image_transforms=None,h=15, w=15):
    image = None
    can_continue = True
    total_time=0
    for m in model.modules():
        
        # The below condition checks for the first 'Conv2d' layer and
        # set the image variable and never gets executed there after. 
        # I assume that there is no 'Conv2d' layer after the 'classifier' layer
        if (isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout) or isinstance(m, nn.ReLU) or \
            isinstance(m, nn.Conv2d)  or isinstance(m, nn.MaxPool2d)  ) and can_continue:
            can_continue = False
            # read a image
            data_dir = 'data/dogImages_crop_350/'
            file = os.path.join(data_dir, 'train/002.Afghan_hound/Afghan_hound_00142.jpg')
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #image = cv2.resize(image,(image_size,image_size))
            #crop_size=int(image_size*0.9)
            #image = crop_around_center(image, crop_size, crop_size)
            print('Original image size: ', image.shape)
            plt.imshow(image)
            plt.show()
            if image_transforms is not None:
                image = F.to_pil_image(image)
                for transform in image_transforms:
                    start = time.time()
                    image = transform(image)
                    end=time.time()-start
                    total_time += end
                    print('-'*127)
                    print('transform: {}  \nimage size: {}  time: {}'.format(transform, np.array(image).shape, end))
                    print('-'*127)
                    plt.imshow(np.array(image))
                    plt.show();
                    
                #image = image.permute(1, 2, 0).numpy() 
                image = np.array(image)
            # show the image

            
            # transfer the x(image) to tensor
            image = transforms.ToTensor()(image)
            image = image.unsqueeze(0)
        
        # Assuming that there should be no layer after classifier
        # layer, I set image to 'None' so that the function 'ShowImgMap'
        # below doesn't get called
        if isinstance(m, nn.Linear) or isinstance(m, nn.Sequential) :
            image = None
        
        #
        if image is not None:
            image, total_time = ShowImgMap(image, m, h, w, total_time)
            
    print('Total time: {}'.format(total_time))
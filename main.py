#!/usr/bin/python

import cv2
import numpy as np
import utils
import time
from debug_module import *
import pickle

# parameters
RATIO = 1/6 # downsampling ratio of the frame
ratio = 1 # downsampling ratio of the background

BG_UPDATE_COEF = 0.95

MEMORY = 5
EVENTS_GAP = 1
TOTAL_EVENTS = 100
MIN_FRAME_COUNT = 16 # min number of frames an object must be present 

FONT = cv2.FONT_HERSHEY_SIMPLEX

morph_size = int(18*RATIO)


# flags
BG_READ_IMAGE = False
BG_DYNAMIC = False
DISP_RESULT = True
SAVE_OBJECTS = True

# ===================================================================

# input video
INPUT_FILENAME =r"Video1.avi"
FILE_IDX = INPUT_FILENAME[-5]
stream = cv2.VideoCapture(INPUT_FILENAME)


# background image
BG_image_name = 'BG' + FILE_IDX + '.jpg'

if BG_READ_IMAGE:    
    BG_image = cv2.imread(BG_image_name)
    # BG_image = cv2.resize(BG_image, None, fx=ratio, fy=ratio)
else:
    ret, BG_image = stream.read()

BG_image_list = []


# background Subtractor
BG_subtractor = cv2.createBackgroundSubtractorMOG2()
#BG_subtractor = cv2.createBackgroundSubtractorKNN()
    


# output video 
OUTPUT_FILENAME ='out_vid' + FILE_IDX + '.avi'
shape = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)*ratio), int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)*ratio)
FPS = stream.get(cv2.CAP_PROP_FPS)
fourcc =cv2.VideoWriter_fourcc(*'XVID')
outputVideo =cv2.VideoWriter(OUTPUT_FILENAME,fourcc,FPS,(shape[1],shape[0]))  


# kernel sizes
morph_elem = cv2.MORPH_ELLIPSE 
elem_size_d = cv2.getStructuringElement(morph_elem , ( 1*morph_size + 1,  1*morph_size + 1))
elem_size_e = cv2.getStructuringElement(morph_elem , ( 1*morph_size + 1,  1*morph_size + 1))

gk_size1 = ( 2*morph_size + 1,  2*morph_size + 1)
gk_size2 = ( 6*morph_size + 1,  6*morph_size + 1)

# initialization
BG_UPDATE_CNT = FPS * 2  # counter for background update
next_id = 0


n_clusters = 4

#blobs = [] 
blobs_old = []
blobs_old_list = []
objects = []
objects_copy = [] 
#tags = []

flag = True  # first blob detection event
start_frame_number = 0
stream.set(1,start_frame_number);

cnt = start_frame_number-1  # frame counter
# ===================================================================
while (True):   

        
        cnt += 1     
        ret, frame_orig = stream.read()
        
        if not ret:
            break
        
        t_start = time.time()    
        frame_orig_res = cv2.resize(frame_orig, None, fx=RATIO, fy=RATIO)
        
                 
        # gaussian blur
        frame = cv2.GaussianBlur(frame_orig, gk_size1, 0)              
        frame_blur = cv2.resize(frame, None, fx=RATIO, fy=RATIO)

        # frame_blur = utils.normalize(frame_blur, 0, 255)
        FG_mask = BG_subtractor.apply(frame_blur)     

          
        # additional processing             
        FG_mask = cv2.GaussianBlur( FG_mask, gk_size2, 0)
        FG_mask = cv2.threshold(FG_mask, 130, 255, 0)[1]      
        # FG_mask = cv2.erode(FG_mask,elem_size_e)
        FG_mask = cv2.dilate(FG_mask,elem_size_d)
        
    
        # cv2.imshow("Foreground", FG_mask)
        blobs_output, blobs_mask = utils.mask_convex_process(FG_mask, frame_orig_res)
        
               
        # update the background
        if  (cnt % BG_UPDATE_CNT == 0 ):
            
            
            frame_orig_res_2 = cv2.resize(frame_orig, None, fx=ratio, fy=ratio)            
            threshold_output_res = cv2.resize(blobs_mask, (frame_orig_res_2.shape[1],frame_orig_res_2.shape[0]))
            output_BG = utils.extract_BG(frame_orig_res_2, threshold_output_res, BG_image, BG_UPDATE_COEF)
            BG_image = output_BG.copy()
            
            # deal with dynamic background
            if BG_DYNAMIC:
                BG_image_list.append(BG_image)



        # extract blobs
        blobs = utils.extract_blobs(blobs_mask, cnt)
        
        if (flag and (len(blobs) > 0) ):
            flag = False
            for j in range(len(blobs)):
                blobs[j].id = next_id
                next_id += 1

        
        elif(len(blobs)> 0):
            
            
            # find ID matches
            tags = utils.id_matches(blobs_old, blobs)
            # tags = utils.id_matches_list(blobs_old_list, blobs, blobs_mask.shape)

            # specify next blob id            
            next_id = utils.update_next_id(blobs, tags, next_id)

      

        if(len(blobs) > 0):
            
            frame_time = cnt/FPS
            
            blobs_id_output = utils.blobs_coloring(blobs, frame_orig_res, frame_time, RATIO)
            
            if DISP_RESULT:
                cv2.imshow("Blobs IDs", blobs_id_output)
                # cv2.imwrite("Blobs_IDs.jpg", blobs_id_output)
                # cv2.imwrite("Frame_curr.jpg", frame_orig)

            # blobs_output = cv2.resize(blobs_output, None, fx=1/RATIO, fy=1/RATIO)            
            # cv2.imshow("blobs", blobs_output)     
            
            objects = utils.create_objects(frame_orig_res, blobs, objects) # create objects from blobs
        
        
        
        blobs_old = blobs.copy() 
        
        # save a list of blobs_old in memory
        if len(blobs_old)>0:        
            if len(blobs_old_list) < MEMORY:
                blobs_old_list.append(blobs_old)
            else:
                del blobs_old_list[0] 
                blobs_old_list.append(blobs_old)
                
          
        
        t_it = time.time() - t_start # time per frame
        label = 'TPF: '+str(round(t_it,3))
        cv2.putText(frame_orig,label,(20,30), FONT, 1.0, (0,255,255),2,cv2.LINE_AA)        
        
        
        # show results
        cv2.imshow("Original frame", frame_orig)
        # cv2.imshow("blobs mask", blobs_mask)

        
        if(cv2.waitKey(1) & 0xff == 27): 
            break
        

 # ===================================================================   
    
 
# save the background 
cv2.imwrite(BG_image_name,BG_image)


# save the objects to disk
if SAVE_OBJECTS:
    save_data = {'objects': objects, 'BG_image_list': BG_image_list}
    # save_data = {'objects': objects}
    filename = 'objects' + FILE_IDX + '.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)   
    
    
    
# remove the objects that are present for less than MIN_FRAME_COUNT frames
objects_copy = objects.copy()
objects_refined = []    

for i in range(len(objects_copy)):           
    if(len(objects_copy[i][1] )> MIN_FRAME_COUNT):
        objects_refined.append(objects_copy[i])
        
    

# draw dominant motion lines
img_lines = utils.motion_lines(objects_refined, BG_image, n_clusters, ratio/RATIO)
GD_image_name = 'img_lines'+ FILE_IDX +'.jpg'
cv2.imwrite(GD_image_name,img_lines)


DISP_METHOD = 'TEXT'

if not BG_DYNAMIC:
    BG_image_list = [BG_image]
    
# summerise the events and save the video  
utils.summerise(objects_refined, BG_image_list, TOTAL_EVENTS, EVENTS_GAP, outputVideo, FPS, ratio/RATIO, stream, DISP_METHOD, BG_UPDATE_CNT)


cv2.waitKey(200)

stream.release()
outputVideo.release()
cv2.destroyAllWindows()

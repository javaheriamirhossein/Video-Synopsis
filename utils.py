#!/usr/bin/python

import cv2
import numpy as np
from mydebug import *
from scipy.stats import itemfreq

# parameters

RATIO = 1/6 # downsampling ratio of the frame
MAX_BLOB_SIZE =int(48000*RATIO)          
MIN_BLOB_SIZE =int(84*RATIO)         
MIN_BLOB_DIST = int(90*RATIO)
MAX_BLOB_DIST = int(180*RATIO)
IOU_THRESH = 0.25

SHOW_TRACK = False # show or hide objects tracks
# ===================================================================
"""
normalize the image to [min_val, max_val]
"""
def normalize(image, min_val, max_val):
    image = np.float32(image)
    max_img = np.max(image)
    min_img = np.min(image)
    image = image-min_img    
    img_norm = np.uint8(image/(max_img-min_img)*(max_val-min_val) + min_val)
    return img_norm


"""
convexify blob contours
- returns the mask output of the frame with the convexhull 
"""
def mask_convex_process(FG_mask, frame_orig_res):
    
    # find contours
    _, contours, hierarchy  = cv2.findContours(FG_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # find the convex hull for each contour
    hull = []
    
    for contour in contours:
        hull.append(cv2.convexHull(contour, False))
    
    
    # draw contours of convex hulls
    for i in range(len(contours)):
        cv2.drawContours(FG_mask, hull, i, (255,255,255),-1, 8)
    
        
    threshold_output = FG_mask
    
    output = cv2.bitwise_and(frame_orig_res, frame_orig_res, mask = threshold_output)
           
    return output, threshold_output



"""
update the background, by merging the new and old images with different coefficients (forget rate)
"""
def extract_BG(frame_orig, mask, current_BG, BG_UPDATE_COEF):

    alfa = BG_UPDATE_COEF    # parameter for updating background
    beta = 1.0 - alfa
    mask_inv = cv2.bitwise_not(mask)
    frame_orig = cv2.bitwise_and(frame_orig, frame_orig, mask = mask_inv)
    updated_BG = cv2.addWeighted(current_BG, alfa, frame_orig, beta, 0)

    return updated_BG




"""
find blob locations
"""
class BlobClass:
    def __init__(self):
        self.blob = [None]
        self.centroid = [0, 0]
        self.id = None
        self.size = None
        self.bbox = None
        self.frame_number = None 
        self.color = None 
        
        

def locate_blobs(mask):
    blobs = []
    
    # assign tags to blobs 
    # 0  - background
    # 1  - untagged foreground
    # 2  - tagged foreground

    
#    flag = True
#    rect_others = [] 
    tagged_mask = np.int32(mask)
    tag = 2     
    
    for y in range(tagged_mask.shape[0]):
        for x in range(tagged_mask.shape[1]):
            if (tagged_mask[y,x] == 1):                               

                _, tagged_mask, _, rect = cv2.floodFill(tagged_mask, None, (x,y), tag)
                        
                blob = []
                
                for i in range(rect[1], rect[1]+rect[3]):      # rect[1]: y and rect[3]: h
                    for  j in range(rect[0], rect[0]+rect[2]): # rect[0]: x and rect[2]: w
                        if (tagged_mask[i,j] == tag):
                            blob.append((j,i))
                                                            
                
                Size = len(blob)
                if ( Size> MIN_BLOB_SIZE and Size< MAX_BLOB_SIZE): # discard very small and very large blobs
                    blobs.append(blob)
                    
#                    if flag:    # first blob detection
#                        blobs.append(blob)
#                        flag = False
#                        
#                    else: # if blobs are too close, reject all except one 
#                        dist_min = MIN_BLOB_DIST+1
#                        for rect_other in rect_others:
#                            dist_corner = np.sqrt( (rect[0]- rect_other[0])**2 + (rect[1]- rect_other[1])**2 )
#                            if dist_corner< dist_min:
#                                dist_min = dist_corner
#                        # if (dist_min > MIN_BLOB_DIST and Size>(MIN_BLOB_SIZE+int(0.5*MIN_BLOB_DIST))):
#                        if (dist_min > MIN_BLOB_DIST):
#                            blobs.append(blob)
#                     
#                    rect_others.append(rect)
                                    
                tag += 1
    
    return blobs
            
    


"""
extract blobs from mask
- blobs_out is a vector of BlobClass blobs
"""
def extract_blobs(mask, frame_number):
    
    blobs_out = []
    binary_mask = cv2.threshold(mask, 0.0, 1.0, cv2.THRESH_BINARY)[1]
    blobs = locate_blobs(binary_mask)


    blob_id = 1
    
    # calculate the center of each blob
    for i in range(len(blobs)):  
        x_temp = 0
        y_temp = 0
        Bc = BlobClass()  
        Bc.blob = blobs[i]
        Bc.size = len(blobs[i])
        Size = len(Bc.blob) 
        
        
        for  j in range(Size):
            x_temp += Bc.blob[j][0]
            y_temp += Bc.blob[j][1]
    
        
        Bc.centroid[0] = x_temp/ Size
        Bc.centroid[1] = y_temp/ Size
        
        Bc.id = blob_id
        Bc.frame_number = frame_number
        blob_id +=1 
        
        blobs_out.append(Bc)            
        
        
    return blobs_out
    



"""
check to see if an old-id has a unique id match
"""
def check_ids(id_old, id_new, centers_dist_min, blob_size, tags):
    
    flag = True
    
    if (id_old != -1):
        for i in  range(len(tags)):
            if (tags[i][0] == id_old ):
                
#                if(centers_dist_min < 0.2*tags[i][2]): # this has been the only actual match
#                    tags[i][2] = centers_dist_min
#                    tags[i][1] = id_new 
#                    flag = False
#                    break
                
                if(centers_dist_min < tags[i][2]): # this has been the only actual match
                    tags[i][2] = centers_dist_min
                    tags[i][1] = id_new 
                flag = False
                break
               
                         
        if (flag):
            if centers_dist_min < MAX_BLOB_DIST:
                if centers_dist_min < blob_size:
                    tag = [id_old, id_new, centers_dist_min]
                    tags.append(tag)            
    
    return tags




"""
find matches between ids and return tags
tags: matches between new and old ids: [blob_id_old, blob_id_new, centers_dist_min]
"""
def id_matches_list(blobs_old_list, blobs_new, mask_shape):
   
    tags = []
    # for each id_new find the nearest id_old
    for i in range(len(blobs_new)): 
        mask = np.zeros(mask_shape)
        for pt in blobs_new[i].blob:
            mask[pt[1],pt[0]] = 1
            
        blobs_new_size = blobs_new[i].size
        
        max_overlap = 0
        max_overlap_id = -1
        
        for blobs_old in blobs_old_list:
            for j in range(len(blobs_old)):
                overlap = 0
                for pt in blobs_old[j].blob:
                    if mask[pt[1],pt[0]] == 1:
                        overlap += 1
                        
                        
                #if overlap> 0.5 *(blobs_old[j].size):        
                if overlap> IOU_THRESH *(blobs_old[j].size + blobs_new_size): # IOU>IOU_THRESH
                    if overlap > max_overlap:
                        max_overlap = overlap
                        max_overlap_id = blobs_old[j].id
             
        if max_overlap_id != -1:  
            tag = [max_overlap_id, blobs_new[i].id, max_overlap]
            tags.append(tag) 
        

    return tags

def id_matches(blobs_old, blobs_new):
   
    tags = []
    # for each id_new find the nearest id_old
    for i in range(len(blobs_new)): 
        centers_dist_min = 2* MAX_BLOB_DIST
        min_dist_blob_old_id = -1
        min_dist_blob_old_size = -1

        
        for j in range(len(blobs_old)):
            dist_centers = np.sqrt( (blobs_new[i].centroid[0] - blobs_old[j].centroid[0])**2 + 
              (blobs_new[i].centroid[1] - blobs_old[j].centroid[1])**2 ) 
            

            if(dist_centers < centers_dist_min):
                centers_dist_min = dist_centers
                min_dist_blob_old_id = blobs_old[j].id
                min_dist_blob_old_size = blobs_old[j].size

        
        
        # check to ensure there is no other id_new associated with the nearest id_old found
        tags = check_ids(min_dist_blob_old_id, blobs_new[i].id, centers_dist_min, min_dist_blob_old_size, tags)
    
    return tags



"""
update the tags of the current blobs with the corresponding tags of the previous frame
and assign new and unique tags to new blobs within the frame
"""
def update_next_id(blobs, tags, next_id):

    
     # The flag "MATCHED" specifies whether a blob is found in matches
    
    for i in range(len(blobs)):
        
        MATCHED = False
        
        for j in range(len(tags)): 
            if(blobs[i].id == tags[j][1]):
                MATCHED = True
                blobs[i].id = tags[j][0]
                break

        
        if not MATCHED:
            blobs[i].id = next_id
            next_id = next_id + 1
            

    return next_id



"""
color the blobs with their object's color
"""
def blobs_coloring(blobs, src_copy, time, Ratio):


    output = np.zeros_like(src_copy)
    
    for i in range(len(blobs)):
        
        blob = blobs[i].blob        
       
        
        # obtain the color of the object in the blob
        
#        pix_color = []
#        for pt in blob: 
#            pix_color.append( src_copy[pt[1],pt[0],:] )
#
#        pix_color = np.float32(pix_color)
#        pixels = pix_color.reshape((-1,3))
#        
#        #number of clusters
#        n_clusters = 2
#        
#        #number of iterations
#        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
#        
#        #initialising centroid
#        
#        #applying k-means to detect prominant color in the image
#        _, labels, centers = cv2.kmeans(pixels, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
#        
#        
#        centers = np.uint8(centers)
#        
#        # dominant cluster  
#        dominant_color = centers[np.argmax(itemfreq(labels)[:, -1])]
#        
#        b = int(dominant_color[0])
#        g = int(dominant_color[1])
#        r = int(dominant_color[2])
        
        
        b = []
        g = []
        r = []
        for pt in blob: 
            b.append( src_copy[pt[1],pt[0],0] )
            g.append( src_copy[pt[1],pt[0],1] )
            r.append( src_copy[pt[1],pt[0],2] )
        
        [hist_b, bin_cent_b] = np.histogram(b, bins=20)
        [hist_g, bin_cent_g] = np.histogram(g, bins=20)
        [hist_r, bin_cent_r] = np.histogram(r, bins=20)
        
        b = np.mean(b)
        g = np.mean(g)
        r = np.mean(r)
        
        alpha = 0.1
        beta = 0.9
        b = alpha*bin_cent_b[np.argmax(hist_b)] + beta*b
        g = alpha*bin_cent_g[np.argmax(hist_g)] + beta*g
        r = alpha*bin_cent_r[np.argmax(hist_r)] + beta*r
        
        blobs[i].color = (int(b),int(g),int(r))

        # paint the blob with this color
        for pt in blob: 
            output[pt[1],pt[0],0] = b
            output[pt[1],pt[0],1] = g
            output[pt[1],pt[0],2] = r

    
    output = cv2.resize(output, None, fx=1/Ratio, fy=1/Ratio)
    
    for i in range(len(blobs)):
    

        blob_id = blobs[i].id
        text = 'id:' + str(blob_id)+':'+str(round(time,2))
        text = 'id:' + str(blob_id)
        scale = 1
        thick = 1

        FONT = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(output, text, (int(blobs[i].centroid[0]/Ratio),int(blobs[i].centroid[1]/Ratio)), FONT, scale, (255,255,255), thick, cv2.LINE_AA)
        
    return output




"""
create a list of identified objects 
"""
def create_objects(src,  blobs, Objects):

    
    for i in range(len(blobs)): 
        
        mask = np.zeros(src.shape[:2], np.uint8)
        blob = blobs[i].blob
        id = blobs[i].id
        
        for pt in blob:
            for j in range(len(src.shape)):
                mask[pt[1],pt[0]] = 1 
                

        blob_attr = [blobs[i].centroid[0], blobs[i].centroid[1], blobs[i].color, blobs[i].frame_number, blobs[i].size]
        
        if(id < len(Objects)):

            
            Objects[id][0].append(blob_attr)
            Objects[id][1].append(mask)
       
        else: 

            while(id >= len(Objects)):
                obj = [[blob_attr],[mask]]
                Objects.append(obj)            

    return Objects
    



"""
draw dominant motion lines
"""
def motion_lines(Objects_ref, BG_image, n_clusters, Ratio):
    
    n_obj = len(Objects_ref)
    
    x_start = [Objects_ref[k][0][0][0]*Ratio for k in range(n_obj)]
    y_start = [Objects_ref[k][0][0][1]*Ratio for k in range(n_obj)]
    
    
    x_end = [Objects_ref[k][0][-1][0]*Ratio for k in range(n_obj)]
    y_end = [Objects_ref[k][0][-1][1]*Ratio for k in range(n_obj)]
    
    
    start_coords = [(x_start[k], y_start[k]) for k in range(n_obj)]
    end_coords = [(x_end[k], y_end[k]) for k in range(n_obj)]


    start_coords = np.float32(start_coords)
    end_coords = np.float32(end_coords)
    
    # clustering criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    
    output = BG_image.copy()
    try:
        # apply k-means to detect centers
        _, start_labels, start_centers = cv2.kmeans(start_coords, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        _, end_labels, end_centers = cv2.kmeans(end_coords, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        line_thick = 1
        tot_samples = len(end_labels)
        
        start_freq =  itemfreq(start_labels)
        for i in range(n_clusters):
            
            max_id = np.argmax(start_freq[:,-1])
            start_freq[max_id,-1] = -1
            
            start_id = start_freq[max_id,0]
            (xs, ys) = start_centers[start_id]
            idx = start_labels == start_id
            
            corresp_end_labels = end_labels[idx]
            if len(corresp_end_labels)> 0.5*tot_samples/n_clusters: # cluster with sufficient number of samples
                
                not_deleted_corresp_labels = corresp_end_labels[ corresp_end_labels > -1]
                
                if len(not_deleted_corresp_labels)> 0:
                    
                    freqs = itemfreq(not_deleted_corresp_labels)
                    match_id = np.argmax(freqs[:, -1])                
                    end_id = freqs[match_id,0]
                
                                    
                    end_labels[end_labels == end_id] = -1 # remove that cluster center
                    (xe, ye) = end_centers[end_id]
                    N_pieces = 100
                    
                    # calculate the slope of the line, delta_x and delta_y
                    delta_x = (xe - xs)/N_pieces
                    delta_y = (ye - ys)/N_pieces
                    theta = np.arctan(-(delta_x/delta_y))
                    w_rect = 0.5
                    delta_x_perp = w_rect* np.cos(theta)
                    delta_y_perp = w_rect* np.sin(theta)
                    delta_color = 255/N_pieces
                    
                    for j in range(N_pieces):
                        color = (int(delta_color*j), int(delta_color*j), int(delta_color*j))
                        # cv2.rectangle(output, (int(xs+j*delta_x-delta_x_perp),int(ys+j*delta_y-delta_y_perp)), (int(xs+(j+1)*delta_x+delta_x_perp),int(ys+(j+1)*delta_y+delta_y_perp)), color, line_thick,-1)
                        for i in range(30):
                            cv2.line(output, (int(xs+j*delta_x-i*delta_x_perp),int(ys+j*delta_y-i*delta_y_perp)), (int(xs+(j+1)*delta_x-i*delta_x_perp),int(ys+(j+1)*delta_y-i*delta_y_perp)), color, line_thick, cv2.LINE_AA)
    
    except Exception:
        pass
    
    return output




"""
substitute the BG_image with the image in thr area of mask 
"""

def substitute(BG_image, mask, stream, frame_number):
    stream.set(1,frame_number);
    ret, image = stream.read()
    if ret:
        image = cv2.resize(image,(BG_image.shape[1],BG_image.shape[0]))
        mask = cv2.resize(mask,(BG_image.shape[1],BG_image.shape[0]))
        mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)[1]
        mask_inv =  cv2.bitwise_not(mask)
        BG_mask = cv2.bitwise_and(BG_image,BG_image,mask = mask_inv)
        img_mask = cv2.bitwise_and(image,image,mask = mask)
        BG_image = cv2.add(BG_mask, img_mask)
    
    return BG_image




"""
summerise the events and save the output video
"""
def summerise(Objects_ref, BG_image_list, TOTAL_EVENTS, EVENTS_GAP, outputVideo, FPS, Ratio, stream, DISP_METHOD, BG_UPDATE_CNT):

    Objects_copy = Objects_ref.copy()
    window_of_obj = []  # window of objects
    grd_window = []     # window of images of objects gradient lines 
    line_colors = []    # color of each part of line connecting object centroids (to create a gradient)
    text_colors = []    # color of text which is the color of object
    line_slopes = []    # slope of line connecting object centroids
    n_frames = []       # number of frames an object is present
    # x_start_vec = []  # starting x coordinate of objects centroids    
    y_start_vec_reg = []# regressed starting y coordinate of objects centroids
    velocity_vec = []   # velocity of objects 
    
    
    # count the numer of frames of each object presence for velocity calculation (by moving averaging)
    counter_vec = []
    
    up_cnt = 0
    down_cnt = 0
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    
    cnt = EVENTS_GAP # farme gap between consecutive objects 
    
    frame_no = 0 # initial farme number of an object 
    
    isList = False
    if len(BG_image_list)>1:
        isList = True
        
    while(Objects_copy):  # continue until there are no objects left 
        
        if isList:
            frame = BG_image_list[int(frame_no/BG_UPDATE_CNT)].copy()
        else:
            frame = BG_image_list[0].copy()
            
        gradient = np.zeros_like(frame)
        mask_grd = np.zeros(gradient.shape[:2], np.uint8)
        
        
        # only add an object to window_of_obj every EVENTS_GAP frames
        if((len(window_of_obj)<TOTAL_EVENTS or len(window_of_obj) < len(Objects_copy)) and (cnt >= EVENTS_GAP)):
            
            # insert an object, set the frame counter to zero
            x_start = Objects_copy[0][0][0][0]*Ratio
            y_start = Objects_copy[0][0][0][1]*Ratio
            size = Objects_copy[0][0][0][4]
            
            # to avoid collision, add an object obly if it is sufficiently distant from others 
            dist_min = 10*size+MIN_BLOB_DIST
            for l in range(len(window_of_obj)):
                x_wind = window_of_obj[l][0][0][0]*Ratio
                y_wind = window_of_obj[l][0][0][1]*Ratio
                dist = np.sqrt( (x_start-x_wind)**2 + (y_start-y_wind)**2 )
                if dist<dist_min:
                    dist_min = dist
                    
            if dist_min > 3*size+MIN_BLOB_DIST:               
        
                window_of_obj.insert(0, Objects_copy[0])
                cnt = 0 
                
                nF = len(Objects_copy[0][0])
                n_frames.insert(0,nF)
                line_colors.insert(0,(0,0,0))
                
                frame_no = Objects_copy[0][0][0][3]
                
                colors = [Objects_copy[0][0][k][2] for k in range(nF)]
                sizes = [Objects_copy[0][0][k][4] for k in range(nF)]
                
                
                # list of colors of object when it is of large size
                max_size = np.max(sizes, axis = 0)
                goosd_size_idx = sizes> 0.8*max_size
                colors_dom = []
                for i in range(nF):
                    if goosd_size_idx[i]:
                        colors_dom.append(colors[i])  
                        
                text_colors.insert(0,colors_dom)
                
                # x_start_vec.insert(0, x_start)
                x_end = Objects_copy[0][0][-1][0]*Ratio
                
                y_start_vec_reg.insert(0, y_start)
                y_end = Objects_copy[0][0][-1][1]*Ratio
                slope = [x_end-x_start, y_end-y_start]
                angle = np.arctan2(slope[1], slope[0]) * 180 / np.pi
                line_slopes.insert(0,slope)
                velocity_vec.insert(0,0)
                counter_vec.insert(0,0)
                
                # specify whether an object is moving up or down
                if angle<0:
                    up_cnt +=1 
                else:
                    down_cnt += 1
                
                
                del Objects_copy[0]                    
                grd_window.insert(0,[gradient, mask_grd])   
                

        
        k = 0
        while (k <len(window_of_obj)):            
#                try:
            
            blob_attr = window_of_obj[k][0][0]
            frame_number = blob_attr[3]

            frame = substitute(frame, window_of_obj[k][1][0], stream, frame_number)                   
            
                        
            # up-down object counter
            scale = 0.8
            thick = 2
            text_up = 'Up: ' + str(up_cnt)
            text_down = 'Down: ' + str(down_cnt)
            cv2.putText(frame, text_up, (20,25), FONT, scale, (0,0,0), thick, cv2.LINE_AA)
            cv2.putText(frame, text_up, (20,25), FONT, scale, (0,255,255), thick-1, cv2.LINE_AA)
            cv2.putText(frame, text_down, (20,55), FONT, scale, (0,0,0), thick, cv2.LINE_AA)
            cv2.putText(frame, text_down, (20,55), FONT, scale, (0,255,255), thick-1, cv2.LINE_AA)
            
            
            # calculate delta x and delta y (normalized by the object size) to obtain velocity 
            # xp = x_start_vec[k]
            xp = blob_attr[0]*Ratio
            yp = y_start_vec_reg[k]
            
            slope = line_slopes[k]
            
            
            if len(window_of_obj[k][0])>1:
                delta_x = (window_of_obj[k][0][1][0]*Ratio - xp)
                delta_y = (window_of_obj[k][0][1][1] - blob_attr[1])*Ratio
                delta_y_reg = delta_x*slope[1]/(slope[0]+1)
                displacement = np.sqrt(delta_x**2 + delta_y**2)/(window_of_obj[k][0][1][4]+10)  # normalize by the size of object
            else:
                delta_x = delta_y = 0
                delta_y_reg = 0
                displacement = 0
            
            

            
            # moving average velocity estimation                               
            velocity_vec[k] += displacement                                        
            counter_vec[k] += 1
            # velocity = velocity_vec[k]/counter_vec[k]*100 
            
            # avoid very large slope
            if abs(slope[1]/slope[0])<40:
                
                # gradually change the color of the line to simulate gradient
                delta_color = int(255/n_frames[k])
                draw_color = line_colors[k]
                line_thick = 10
                cv2.line(grd_window[k][0], (int(xp),int(yp)), (int(xp+delta_x),int(yp+delta_y_reg)), draw_color, line_thick)
                cv2.line(grd_window[k][1], (int(xp),int(yp)), (int(xp+delta_x),int(yp+delta_y_reg)), (255,255,255), line_thick)
                
                line_colors[k] = (draw_color[0]+delta_color, draw_color[1]+delta_color, draw_color[2]+delta_color)
                # x_start_vec[k] = xp+delta_x
                y_start_vec_reg[k] = yp+delta_y_reg

            
            del window_of_obj[k][0][0] 
            del window_of_obj[k][1][0] 


            
            if not (window_of_obj[k][0]):
                del window_of_obj[k]
                del line_colors[k]
                del line_slopes[k]
                del n_frames[k]
                # del x_start_vec[k]
                del y_start_vec_reg[k]
                del grd_window[k]
                del velocity_vec[k]
                del counter_vec[k]
                
            k +=1 
        
#                except Exception:
#                    pass
            
        if SHOW_TRACK:
            gradient_img = np.zeros_like(frame)
            mask_grd_img = np.zeros(gradient_img.shape[:2], np.uint8)
            
            for ct in range(len(window_of_obj)):
                gradient_img = cv2.add(gradient_img, grd_window[ct][0])
                mask_grd_img = cv2.add(mask_grd_img, grd_window[ct][1])
            
            mask_inv = cv2.bitwise_not(mask_grd_img)   
            frame = cv2.bitwise_and(frame,frame,mask = mask_inv)        
            frame = cv2.add(frame,gradient_img)
        
        
        
        for ct in range(len(window_of_obj)): 
                        
            
            blob_attr = window_of_obj[ct][0][0]
            txt_color = np.mean(text_colors[ct], axis=0)
            # txt_color = text_colors[ct]
            
            # color of the object
            color = (int(txt_color[0]), int(txt_color[1]), int(txt_color[2]))
            
            scale = 0.6
            
            velocity = velocity_vec[ct]/counter_vec[ct]*100 
            text_v = 'v:'+str(round(velocity,2))    # velocity
            text_t = 't:'+str(round(blob_attr[3]/FPS / 60,2)) # time
            
            # display the time of object presence and velocity as text

            
            if DISP_METHOD == 'RECT':
                
                # texts are shown in a rectangle whose color is the color of the object 
                thick = 2
                cv2.rectangle(frame, (int(blob_attr[0]*Ratio)-5,int(blob_attr[1]*Ratio)-30), (int(blob_attr[0]*Ratio)+55,int(blob_attr[1]*Ratio)+5), color,-1)
                cv2.putText(frame, text_t, (int(blob_attr[0]*Ratio),int(blob_attr[1]*Ratio)), FONT, scale, (255,255,255), thick-1,cv2.LINE_AA)
                                                 
                                 
                thick = 2  
                cv2.putText(frame, text_v, (int(blob_attr[0]*Ratio),int(blob_attr[1]*Ratio)-15), FONT, scale, (255,255,255), thick-1,cv2.LINE_AA)       
            
            else:
                
                # the color of text is the color of the object                
                thick = 3                         
                cv2.putText(frame, text_t, (int(blob_attr[0]*Ratio),int(blob_attr[1]*Ratio)), FONT, scale, (0,0,0), thick,cv2.LINE_AA)
                cv2.putText(frame, text_t, (int(blob_attr[0]*Ratio),int(blob_attr[1]*Ratio)), FONT, scale, color, thick-1,cv2.LINE_AA)
                                
                
                thick = 3                        
                cv2.putText(frame, text_v, (int(blob_attr[0]*Ratio),int(blob_attr[1]*Ratio)-15), FONT, scale, (0,0,0), thick, cv2.LINE_AA)
                cv2.putText(frame, text_v, (int(blob_attr[0]*Ratio),int(blob_attr[1]*Ratio)-15), FONT, scale, color, thick-1, cv2.LINE_AA)       

           
        cv2.imshow("Summerized video",frame)


        cnt +=1 

        # save the video
        if not outputVideo.isOpened():        
            print("Cannot save video \n")
            return        
        else:
            outputVideo.write(frame) 
        
        
        if(cv2.waitKey(1) & 0xff ==27): 
            break
        
import cv2
import skimage as ski
import numpy as np
from lxml import etree
import os

PATH_RAW_VIDEOS = "/home/natalia/smoke_workspace/grid_module/data/raw_videos/"
PATH_CVAT_LABELS = "/home/natalia/smoke_workspace/grid_module/data/cvat_label/"

def split_labels(xml_filename, img_width, img_height):
    KNOWN_TAGS = {'box', 'image', 'attribute'}
    print(xml_filename)
    cvat_xml = etree.parse(xml_filename)
    tracks= cvat_xml.findall( './/track' )

    sizes=cvat_xml.findall( './/original_size' )
    for size in sizes:
        width = size.find('width').text
        height = size.find('height').text


    original_height = int(height)
    original_width = int(width)


    frames = {}

    for track in tracks:
        trackid = int(track.get("id"))
        label = track.get("label")
        boxes = track.findall( './box' )
        for box in boxes:
            frameid  = int(box.get('frame'))
            outside  = int(box.get('outside'))
            xtl      = float(box.get('xtl'))*img_width/original_width
            ytl      = float(box.get('ytl'))*img_height/original_height
            xbr      = float(box.get('xbr'))*img_width/original_width
            ybr      = float(box.get('ybr'))*img_height/original_height
            
            frame = frames.get( frameid, {} )
            
            if outside == 0:
                frame[ trackid ] = { 'xtl': xtl, 'ytl': ytl, 'xbr': xbr, 'ybr': ybr, 'label': label }
                
            frames[frameid] = frame

    return frames

def optical_flow(img, prev):
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(img)
    hsv[..., 1] = 255
    next = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp_next = ski.feature.local_binary_pattern(next, 8,1, method="uniform")
    
    flow = cv2.calcOpticalFlowFarneback(prev, next, flow=None, pyr_scale=0.5,
                                        levels=3, winsize=20, iterations=1,
                                        poly_n=1, poly_sigma=1.1, flags=0)
   
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] =cv2.normalize(lbp_next, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr

def make_txt_bbox_file(bbox_info, output_file):
    objids = sorted(bbox_info.keys())
    with open(output_file, 'w') as file:
        for objid in objids:
            box = bbox_info[objid]
            xmin = int(box.get('xtl'))
            ymin = int(box.get('ytl'))
            xmax = int(box.get('xbr'))
            ymax = int(box.get('ybr'))

            # cv2.rectangle(img, (xmin,ymin),(xmax,ymax), (255,0,0))
            # cv2.imshow("frame",img)
            # cv2.waitKey(0)

            file.write(str(xmin)+","+str(ymin)+","+str(xmax)+','+str(ymax)+"\n")

    
def split_video(input_video, output_folder, max_frames=50, frame_gap=5, img_width=640, img_height=480):
    if not os.path.exists(output_folder+"/frame"):
        os.makedirs(output_folder+"/frame")
    if not os.path.exists(output_folder+"/prev_frame"):
        os.makedirs(output_folder+"/prev_frame")
    if not os.path.exists(output_folder+"/opt_flow_frame"):
        os.makedirs(output_folder+"/opt_flow_frame")
    if not os.path.exists(output_folder+"/bbox_annotation"):
        os.makedirs(output_folder+"/bbox_annotation")

    print(input_video)
    vidcap = cv2.VideoCapture(PATH_RAW_VIDEOS+input_video)
    success = True
    frame, saved_frames_count = 0, 0
    frames=split_labels(PATH_CVAT_LABELS+input_video+".xml", img_width, img_height)
    init = True

    while saved_frames_count < max_frames:
        success, img = vidcap.read()
        img=cv2.resize(img,(img_width,img_height))
        frame += 1
        if not success:
                break
        if init:
            prev = img
            init=False

        if frame % frame_gap == 0:
            jpg_name = input_video + '_frame_' + str(frame) + '.jpg'
            opt_image = optical_flow(img,prev)

            cv2.imwrite(output_folder+"/frame" + '/' + jpg_name, img)
            cv2.imwrite(output_folder+"/prev_frame" + '/' + jpg_name, prev)
            cv2.imwrite(output_folder+"/opt_flow_frame" + '/' + jpg_name, opt_image)
            saved_frames_count += 1

            if frame in frames.keys():
                make_txt_bbox_file(frames[frame], output_folder+"/bbox_annotation" + '/' + jpg_name+".txt")
            else:
                f=open(output_folder+"/bbox_annotation" + '/' + jpg_name+".txt", 'w')
                f.close()
   
    vidcap.release()
    print(f'Got {saved_frames_count} frames from {input_video}')


if __name__ == '__main__':
    train_file = "/home/natalia/smoke_workspace/grid_module/data/train_files.txt"
    test_file = "/home/natalia/smoke_workspace/grid_module/data/test_files.txt"
    val_file = "/home/natalia/smoke_workspace/grid_module/data/val_files.txt"


    with open(train_file, 'r') as file:
        video_names = file.read().splitlines() 
        for video_name in video_names:
            split_video(input_video=video_name, 
                        output_folder="data/image_dataset/train", max_frames=100, frame_gap=10)

    with open(test_file, 'r') as file:
        video_names = file.read().splitlines() 
        for video_name in video_names:
            split_video(input_video=video_name, 
                        output_folder="data/image_dataset/test", max_frames=100, frame_gap=10)


    with open(val_file, 'r') as file:
        video_names = file.read().splitlines() 
        for video_name in video_names:
            split_video(input_video=video_name, 
                        output_folder="data/image_dataset/val", max_frames=100, frame_gap=10)
        
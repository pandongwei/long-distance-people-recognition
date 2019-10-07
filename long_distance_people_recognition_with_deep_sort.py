#!/usr/bin/env python3

import time
from mtcnn.src import detect_faces, show_bboxes
from torch import torch
from ArcFace.mobile_model import mobileFaceNet
from util_face_recognition import cosin_metric, get_feature, draw_ch_zn
import numpy as np
import cv2
import os
from torchvision import transforms
from PIL import Image,ImageFont

#from yolov3
import time
import torch.nn as nn
from torch.autograd import Variable
from util_people_detection import *
from darknet import Darknet
from preprocess import  inp_to_image
import random 
import argparse
import pickle as pkl

#from kalman filter
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
from deep_sort.detection import Detection as ddet

'''
#preparation about realsense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
# Start streaming
profile = pipeline.start(config)
'''
#parameters from face recognition
font = ImageFont.truetype('simhei.ttf',20,encoding='utf-8')
cfgfile = "cfg/yolov3.cfg"
weightsfile = "yolov3.weights"
num_classes = 80
#parameters from people detection
classes = load_classes('data/coco.names')
colors = pkl.load(open("pallete", "rb"))
#parameters from kalman filter
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0

#functions from yolov3
def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 

    img_是适配后的图像
    orig_im是原始图像
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def get_test_input(input_dim, CUDA):
    img = cv2.imread("imgs/messi.jpg")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    
    return img_

def write(x, img):
    c1 = tuple(x[1:3].astype(int))
    c2 = tuple(x[3:5].astype(int))
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.25)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "160", type = str)
    return parser.parse_args()

def sellect_person(output):
    '''
    筛选output，只有标签为person的才被保留
    '''
    result = []
    for i in output:
        if i[-1] == 0:
            result.append(i)
    return result

def to_tlwh(outputs):                     

    output_tlwh = []
    for output in outputs:
        t = int(output[1])
        l = int(output[2])
        w = int(output[3] - output[1])
        h = int(output[4] - output[2])
        output_tlwh.append([t,l,w,h])
    return output_tlwh

def main():
##########################################################################################################
    #preparation part
    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()
    
    num_classes = 80
    
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    
    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    
    assert inp_dim % 32 == 0                   #assert后面语句为false时触发，中断程序
    assert inp_dim > 32

    if CUDA:
        model.cuda()
            
    model.eval()
    
    global confirm
    global person
    
    fps = 0.0
    count = 0
    frame = 0    
    person = []
    confirm = False
    reconfirm = False
    count_yolo = 0
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1) 
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    #record the video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('output/testwrite_normal.avi',fourcc, 15.0, (640,480),True)

    cap = cv2.VideoCapture(0)

    detect_time = []
    recogn_time = []
    kalman_time = []
    aux_time = []
    while True:
        start = time.time()  
        ret, color_image = cap.read()
        '''
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        '''
        if color_image is None:
            break
        img, orig_im, dim = prep_image(color_image, inp_dim)
        
        im_dim = torch.FloatTensor(dim).repeat(1,2)             
##########################################################################################################
        #people detection part                
        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()
        time_a = time.time()
        if count_yolo %3 == 0:                                                               #detect people every 3 frames
            output = model(Variable(img), CUDA)                         #适配后的图像放进yolo网络中，得到检测的结果
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)         


            if type(output) == int:
                fps  = ( fps + (1./(time.time()-start)) ) / 2
                print("fps= %f"%(fps))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
        
            output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim                #夹紧张量，限制在一个区间内
        
            #im_dim = im_dim.repeat(output.size(0), 1)
            output[:,[1,3]] *= color_image.shape[1]
            output[:,[2,4]] *= color_image.shape[0]
            output = output.cpu().numpy() 
            output = sellect_person(output)                                       #把标签不是人的output去掉，减少计算量
            output = np.array(output)
            output_update = output
        elif count_yolo %3 != 0:
            output = output_update
        count_yolo += 1
        list(map(lambda x: write(x, orig_im), output))                #把结果加到原来的图像中   
        #output的[0,1:4]分别为框的左上和右下的点的位置
        detect_time.append(time.time() - time_a)
##########################################################################################################
        time_a = time.time()
        #kalman filter part
        outputs_tlwh = to_tlwh(output)                             ##把output数据变成适合kalman更新的类型
        features = encoder(orig_im,outputs_tlwh)
        detections = [Detection(output_tlwh, 1.0, feature) for output_tlwh, feature in zip(outputs_tlwh, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            box = track.to_tlbr()
            cv2.rectangle(orig_im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),(255,255,255), 2)
            cv2.putText(orig_im, str(track.track_id),(int(box[0]), int(box[1])),0, 5e-3 * 200, (0,255,0),2)  
        
        kalman_time.append(time.time() - time_a)
##########################################################################################################
        #face recognition part
        time_a = time.time()
        if confirm == False:
            saved_model = './ArcFace/model/068.pth'
            name_list = os.listdir('./users')
            path_list = [os.path.join('./users',i,'%s.txt'%(i)) for i in name_list]
            total_features = np.empty((128,),np.float32)

            for i in path_list:
                temp = np.loadtxt(i)
                total_features = np.vstack((total_features,temp))
            total_features = total_features[1:]

            #threshold = 0.30896     #阈值并不合适，可能是因为训练集和测试集的差异所致！！！
            threshold = 0.5
            model_facenet = mobileFaceNet()
            model_facenet.load_state_dict(torch.load(saved_model)['backbone_net_list'])
            model_facenet.eval()
            #use_cuda = torch.cuda.is_available() and True
            #device = torch.device("cuda" if use_cuda else "cpu")
            device = torch.device("cuda")

            # is_cuda_avilable
            trans = transforms.Compose([
                transforms.Resize((112,112)),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            ])
            model_facenet.to(device)

            img = Image.fromarray(color_image)
            bboxes, landmark = detect_faces(img)                                                                  #首先检测脸

            if len(bboxes) == 0:
                print('detect no people')
            else:
                for bbox in bboxes:
                    loc_x_y = [bbox[2], bbox[1]]
                    person_img = color_image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])].copy()              #从图像中截取框
                    feature = np.squeeze(get_feature(person_img, model_facenet, trans, device))                               #框里的图像计算feature
                    cos_distance = cosin_metric(total_features, feature)
                    index = np.argmax(cos_distance)
                    if  cos_distance[index] <= threshold:
                        continue
                    person = name_list[index]  
                    #在这里加框加文字
                    orig_im = draw_ch_zn(orig_im,person,font,loc_x_y)                                                                    #加名字
                    cv2.rectangle(orig_im,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))           #加box
            #cv2.imshow("frame", orig_im)

##########################################################################################################
            #confirmpart
            print('confirmation rate: {} %'.format(count*10))
            cv2.putText(orig_im, 'confirmation rate: {} %'.format(count*10), (10,30),cv2.FONT_HERSHEY_PLAIN, 2, [0,255,0], 2)
            if len(bboxes)!=0 and len(output)!=0:
                if bboxes[0,0]>output[0,1] and bboxes[0,1]>output[0,2] and bboxes[0,2]<output[0,3] and bboxes[0,3]<output[0,4] and person:
                    count+=1
            frame+=1
            if count>=10 and frame<=30:
                confirm = True
                print('confirm the face is belong to that people')
            elif  frame >= 30:
                print('fail confirm, and start again')
                reconfirm = True
                count = 0
                frame = 0
            if reconfirm == True:
                cv2.putText(orig_im, 'fail confirm, and start again', (10,60),cv2.FONT_HERSHEY_PLAIN, 2, [0,255,0], 2)                   
##########################################################################################################
        recogn_time.append(time.time() - time_a)
        time_a = time.time()
        #show the final output result
        if not confirm:
            cv2.putText(orig_im, 'still not confirm', (output[0,1].astype(np.int32)+100,output[0,2].astype(np.int32)+20),
                                     cv2.FONT_HERSHEY_PLAIN, 2, [0,0,255], 2)
        #把识别的名字加上去
        if confirm:  
            for track in tracker.tracks:
                bbox = track.to_tlbr()
                if track.track_id == 1:
                    cv2.putText(orig_im, person, (int(bbox[0])+100,int(bbox[1])+20),
                                            cv2.FONT_HERSHEY_PLAIN, 2, [0,255,0], 2)
                
                    #rate.sleep()
        cv2.imshow("frame", orig_im)
        #out.write(orig_im)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        
        aux_time.append(time.time()-time_a)
        fps  = ( fps + (1./(time.time()-start)) ) / 2
        print("fps= %f"%(fps))
    #calculate how long each part takes
    avg_detect_time = np.mean(detect_time)
    avg_recogn_time = np.mean(recogn_time)
    avg_kalman_time = np.mean(kalman_time)
    avg_aux_time = np.mean(aux_time)
    print("avg detect: {}".format(avg_detect_time))
    print("avg recogn: {}".format(avg_recogn_time))
    print("avg kalman: {}".format(avg_kalman_time))
    print("avg aux: {}".format(avg_aux_time))
    print("avg fps: {}".format(1/(avg_detect_time + avg_recogn_time + avg_kalman_time + avg_aux_time)))

if __name__ =='__main__':
    main()

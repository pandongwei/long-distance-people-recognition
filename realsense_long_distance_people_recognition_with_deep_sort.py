'''
两个网络融合
只检测人，并框出来
当人脸检测框在人物检测框里面时，把识别到的人的名字转移到人物检测框上面
confirm过程可视化
解除confirm时len=1的限制
加上kalman filter,第二种
用realsense作为输入，显示距离信息和去掉背景
'''

from __future__ import division
import time
import pyrealsense2 as rs
from cv2 import cv2
import os
import numpy as np
from torch import torch
#from face recognition
from mtcnn.src import detect_faces, show_bboxes
from torch import torch
from ArcFace.mobile_model import mobileFaceNet
from util_face_recognition import cosin_metric, get_feature, draw_ch_zn
from torchvision import transforms
from PIL import Image,ImageFont,ImageDraw

#from yolov3
import time
import torch.nn as nn
from torch.autograd import Variable
from util_people_detection import *
from darknet import Darknet
from preprocess import  inp_to_image
import pandas as pd
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

def get_dist_info(depth_image,bounding_bbox_position):
    #  只取depth_image中的框中最中间的小框，进行深度的计算，然后平均，确保计算的框中的像素都是人脸而不包括远距离的背景
    '''
    #框的边长是bounding box的0.5倍
    depth_image = depth_image[int(0.75*bounding_bbox_position[0] + 0.25*bounding_bbox_position[2]):int(0.25*bounding_bbox_position[0] + 0.75*bounding_bbox_position[2]),
                                                                int(0.75*bounding_bbox_position[1] + 0.25*bounding_bbox_position[3]):int(0.25*bounding_bbox_position[1] + 0.75*bounding_bbox_position[3])].astype(float)
    '''
    '''
    #框的边长是bounding box的1/3
    depth_image = depth_image[int(0.83*bounding_bbox_position[0] + 0.17*bounding_bbox_position[2]):int(0.17*bounding_bbox_position[0] + 0.83*bounding_bbox_position[2]),
                                                                int(0.83*bounding_bbox_position[1] + 0.17*bounding_bbox_position[3]):int(0.17*bounding_bbox_position[1] + 0.83*bounding_bbox_position[3])].astype(float) 
    '''
    #框只取中间的几个pixels
    depth_image = depth_image[int(0.5*(bounding_bbox_position[0] + bounding_bbox_position[2]))-1:int(0.5*(bounding_bbox_position[0] + bounding_bbox_position[2]))+1,
                                                                int(0.5*(bounding_bbox_position[1] + bounding_bbox_position[3]))-1:int(0.5*(bounding_bbox_position[1] + bounding_bbox_position[3]))+1].astype(float)  
   
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()             #这得到的是一个常数
    depth = depth_image * depth_scale
    dist,_,_,_ = cv2.mean(depth)              #深度平均一下
    return dist
'''
def clip_rest(color_image,depth_image,dist):
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    clipping_distance_front= (dist-0.5) / depth_scale
    clipping_distance_back = (dist+0.5) / depth_scale
    grey_color = 153
    depth_image_3d = np.dstack((depth_image,depth_image,depth_image))
    color_image = np.where((depth_image_3d > clipping_distance_back) | (depth_image_3d < clipping_distance_front) , grey_color, color_image)
    
    return color_image
'''
def add_dist_info(color_image,bounding_bbox_position,dist):
    # 图像从OpenCV格式转换成PIL格式
    img_PIL = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    # 字体  
    font = ImageFont.truetype('FreeMonoBoldOblique.ttf', 30, encoding="utf-8")
    # 开始显示
    draw = ImageDraw.Draw(img_PIL)
    draw.text((int(bounding_bbox_position[0]+30), int(bounding_bbox_position[1]-30)), "face distance  " + str(round(dist,2)), font=font, fill=(255,0,0))

    # 转换回OpenCV格式
    img_OpenCV = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)

    return img_OpenCV    


#########################################################################################################
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

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)
# Start streaming
profile = pipeline.start(config)
##########################################################################################################

def main():
##########################################################################################################
    #preparation part
    '''
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    profile = pipeline.start(config)
    '''
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

    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1) 
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    #record the video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output/testwrite_realsense1.avi',fourcc, 14.0, (640,480),True)

    #cap = cv2.VideoCapture(0)
    while True:
        start = time.time()  
        #ret, color_image = cap.read()
        
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        #使用各种Filter来改善深度图的质量
        #降低解释度的Filter但是使画面更加精细
        decimation = rs.decimation_filter()
        decimation.set_option(rs.option.filter_magnitude, 2)
        #decimation.set_option(rs.option.accuracy,2)
        decimated_depth = decimation.process(depth_frame)

        #可以使画面平滑的filter
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 5)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial.set_option(rs.option.filter_smooth_delta, 20)
        spatial.set_option(rs.option.holes_fill, 3)
        filtered_depth = spatial.process(decimated_depth)

        #填补空洞的filter
        hole_filling = rs.hole_filling_filter()
        hole_filling.set_option(rs.option.holes_fill,2)
        filled_depth = hole_filling.process(filtered_depth)      

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(filled_depth.get_data())       

        img, orig_im, dim = prep_image(color_image, inp_dim)
        
        im_dim = torch.FloatTensor(dim).repeat(1,2)             
##########################################################################################################
        #people detection part                
        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()
        
        
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
        list(map(lambda x: write(x, orig_im), output))                #把结果加到原来的图像中   
        #output的[0,1:4]分别为框的左上和右下的点的位置
##########################################################################################################
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
##########################################################################################################
        #face recognition part

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
            cv2.putText(orig_im, 'confirmation rate: {} %'.format(count*2.5), (10,30),cv2.FONT_HERSHEY_PLAIN, 2, [0,255,0], 2)
            if len(bboxes)!=0 and len(output)!=0:
                if bboxes[0,0]>output[0,1] and bboxes[0,1]>output[0,2] and bboxes[0,2]<output[0,3] and bboxes[0,3]<output[0,4] and person:
                    count+=1
            frame+=1
            if count>=40 and frame<=100:
                confirm = True
                print('confirm the face is belong to that people')
            elif  frame >= 100:
                print('fail confirm, and start again')
                reconfirm = True
                count = 0
                frame = 0
            if reconfirm == True:
                cv2.putText(orig_im, 'fail confirm, and start again', (10,60),cv2.FONT_HERSHEY_PLAIN, 2, [0,255,0], 2)                   
##########################################################################################################
        #show the final output result
        if not confirm:
            cv2.putText(orig_im, 'still not confirm', (output[0,1].astype(np.int32)+100,output[0,2].astype(np.int32)+20),
                                     cv2.FONT_HERSHEY_PLAIN, 2, [0,0,255], 2)
        if confirm:
            for track in tracker.tracks:
                bbox = track.to_tlbr()
                if track.track_id == 1:
                    cv2.putText(orig_im, person, (int(bbox[0])+100,int(bbox[1])+20),
                                            cv2.FONT_HERSHEY_PLAIN, 2, [0,255,0], 2)
                    dist_info = get_dist_info(depth_image,bbox)
                    #orig_im = clip_rest(color_image,depth_image,dist_info)
                    orig_im = add_dist_info(orig_im,bbox,dist_info)

        cv2.imshow("frame", orig_im)
        out.write(orig_im)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        
        fps  = ( fps + (1./(time.time()-start)) ) / 2
        print("fps= %f"%(fps))

if __name__ =='__main__':
    main()

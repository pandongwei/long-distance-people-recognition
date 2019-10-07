import pyrealsense2 as rs
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)
profile = pipeline.start(config)
#from face recognition
import time
from mtcnn.src import detect_faces, show_bboxes
import torch
from ArcFace.mobile_model import mobileFaceNet
from util_face_recognition import cosin_metric, get_feature, draw_ch_zn
import numpy as np
import cv2
import os
from torchvision import transforms
from PIL import Image,ImageFont,ImageDraw

#from yolov3
import time
import torch.nn as nn
from torch.autograd import Variable
from util_people_detection import *
from darknet import Darknet
from preprocess import  inp_to_image
#import pandas as pd
import random 
import argparse
import pickle as pkl

#from Kalman filter
from kalman_filter.detectors import Detectors
from kalman_filter.tracker import Tracker


#parameters from face recognition
font = ImageFont.truetype('simhei.ttf',20,encoding='utf-8')
cfgfile = "cfg/yolov3.cfg"
weightsfile = "yolov3.weights"
num_classes = 80
#parameters from people detection
classes = load_classes('data/coco.names')
colors = pkl.load(open("pallete", "rb"))

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

def to_xy(outputs):
    '''
    把output的格式变为2X1X人物数的多维矩阵
    x,y为框的中心点
    '''
    output_xy = []
    for output in outputs:
        x = 0.5*(output[1] + output[3])
        y = 0.5*(output[2] + output[4])
        output_xy.append([[x],[y]])
    return output_xy

def xy_to_normal(outputs,tracks):
    '''
    中心点位置更新过后，将其换回原来的数据形式
    '''
    output_normal = []
    i = 0
    for output in outputs:
        x_l = int(tracks[i].prediction[0] - 0.5*(output[3]-output[1]))
        y_l = int(tracks[i].prediction[1] - 0.5*(output[4]-output[2]))
        x_r = int(tracks[i].prediction[0] + 0.5*(output[3]-output[1]))
        y_r = int(tracks[i].prediction[1] + 0.5*(output[4]-output[2]))
        id = tracks[i].track_id
        output_normal.append([x_l,y_l,x_r,y_r,id])
        i+=1
    return output_normal

def get_dist_info(depth_image,bounding_bbox_position):
    #  只取depth_image中的框中最中间的小框，进行深度的计算，然后平均，确保计算的框中的像素都是人脸而不包括远距离的背景
    '''
    #框的边长是bounding box的0.5倍
    depth_image = depth_image[int(0.75*bounding_bbox_position[0] + 0.25*bounding_bbox_position[2]):int(0.25*bounding_bbox_position[0] + 0.75*bounding_bbox_position[2]),
                                                                int(0.75*bounding_bbox_position[1] + 0.25*bounding_bbox_position[3]):int(0.25*bounding_bbox_position[1] + 0.75*bounding_bbox_position[3])].astype(float)
    '''
    
    #框的边长是bounding box的1/3
    depth_image = depth_image[int(0.83*bounding_bbox_position[0] + 0.17*bounding_bbox_position[2]):int(0.17*bounding_bbox_position[0] + 0.83*bounding_bbox_position[2]),
                                                                int(0.83*bounding_bbox_position[1] + 0.17*bounding_bbox_position[3]):int(0.17*bounding_bbox_position[1] + 0.83*bounding_bbox_position[3])].astype(float) 
    '''
    #框只取中间的几个pixels
    depth_image = depth_image[int(0.5*(bounding_bbox_position[0] + bounding_bbox_position[2]))-5:int(0.5*(bounding_bbox_position[0] + bounding_bbox_position[2]))+5,
                                                                int(0.5*(bounding_bbox_position[1] + bounding_bbox_position[3]))-5:int(0.5*(bounding_bbox_position[1] + bounding_bbox_position[3]))+5].astype(float)  
   '''
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()             #这得到的是一个常数
    depth = depth_image * depth_scale
    dist = []
    for i in range(len(depth)):
        for j in range(len(depth[0])):
            if depth[i,j] != 0:
                dist.append(depth[i,j])
    dist = sum(dist)/(len(dist)+1)
    #dist,_,_,_ = cv2.mean(depth)              #深度平均一下
    return dist


def add_dist_info(color_image,bounding_bbox_position,dist):
    # 图像从OpenCV格式转换成PIL格式
    img_PIL = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    # 字体  
    font = ImageFont.truetype('FreeMonoBoldOblique.ttf', 30, encoding="utf-8")
    # 开始显示
    draw = ImageDraw.Draw(img_PIL)
    draw.text((int(bounding_bbox_position[0]+10), int(bounding_bbox_position[1]+40)), "distance  " + str(round(dist,2)), font=font, fill=(255,0,0))

    # 转换回OpenCV格式
    img_OpenCV = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)

    return img_OpenCV    

def main():
##########################################################################################################
    #preparation part
    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()
    
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    
    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    
    assert inp_dim % 32 == 0                   #assert后面语句为false时触发，中断程序
    assert inp_dim > 32
    
    if CUDA:
        model.cuda()
   
    model.eval()

    #Kalman Filter
    tracker = Tracker(dist_thresh = 160, max_frames_to_skip = 100, 
                                        max_trace_length = 5, trackIdCount = 1)
    
    global confirm
    global person
    
    fps = 0.0
    count = 0
    frame = 0    
    person = []
    confirm = False
    reconfirm = False
    count = 0                       #每3帧进行一次检测，更新人物位置，其余情况下位置保持不变

    #record the video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output/output_kalman111.avi',fourcc, 18.0, (640,360),True)

    #cap = cv2.VideoCapture(0)

    while True:
        start = time.time()  
        align_to = rs.stream.color
        align = rs.align(align_to)
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        #ret, color_image = cap.read()

        #可以使画面平滑的filter
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 5)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial.set_option(rs.option.filter_smooth_delta, 20)
        spatial.set_option(rs.option.holes_fill, 3)
        filtered_depth = spatial.process(depth_frame)
        
        #填补空洞的filter
        hole_filling = rs.hole_filling_filter()
        hole_filling.set_option(rs.option.holes_fill,2)
        filled_depth = hole_filling.process(filtered_depth)      
        
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(filled_depth.get_data())   
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        img, orig_im, dim = prep_image(color_image, inp_dim)
        
        im_dim = torch.FloatTensor(dim).repeat(1,2)  
                
##################################################################################################
        #people detection part                
        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()
        
        if count %3 == 0:
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
        elif count%1 !=0:
            output = output_update
        count +=1
        #list(map(lambda x: write(x, orig_im), output))                #把结果加到原来的图像中   
        #output的[0,1:4]分别为框的左上和右下的点的位置
###########################################################################################################
        #kalman filter tracking part

        output_kalman_xywh = to_xy(output)                   #把output数据变成适合kalman更新的类型
        if (len(output_kalman_xywh) > 0):
            tracker.Update(output_kalman_xywh)                #用kalman filter更新框的位置
        
        outputs_kalman_normal = np.array(xy_to_normal(output,tracker.tracks)) #换回原来的数据形式
        #画框
        for output_kalman_normal in outputs_kalman_normal:
            cv2.rectangle(orig_im, (int(output_kalman_normal[0]), int(output_kalman_normal[1])), 
                                        (int(output_kalman_normal[2]), int(output_kalman_normal[3])),(255,255,255), 2)
            cv2.rectangle(depth_colormap, (int(output_kalman_normal[0]), int(output_kalman_normal[1])), 
                                        (int(output_kalman_normal[2]), int(output_kalman_normal[3])),(255,255,255), 2)
            cv2.putText(orig_im, str(output_kalman_normal[4]),(int(output_kalman_normal[0]), int(output_kalman_normal[1])),
                                    0, 5e-3 * 200, (0,255,0),2)              #track id 就是数字  

#tracker.tracks[i].track_id
########################################################################################################
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
############################################################################################################
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

###############################################################################################################
        #show the final output result
        if not confirm:
            cv2.putText(orig_im, 'still not confirm', (output[0,1].astype(np.int32)+100,output[0,2].astype(np.int32)+20),
                                     cv2.FONT_HERSHEY_PLAIN, 2, [0,0,255], 2)
        if confirm:
            for output_kalman_normal in outputs_kalman_normal:
                if output_kalman_normal[4] == 1:
                    cv2.putText(orig_im, person, (output_kalman_normal[0].astype(np.int32)+100,output_kalman_normal[1].astype(np.int32)+20),
                                            cv2.FONT_HERSHEY_PLAIN, 2, [0,255,0], 2)
                    dist_info = get_dist_info(depth_image,output_kalman_normal)
                    #orig_im = clip_rest(color_image,depth_image,dist_info)
                    #depth_colormap = add_dist_info(depth_colormap,bbox,dist_info)
                    orig_im = add_dist_info(orig_im,output_kalman_normal,dist_info)
        #images = np.hstack((orig_im, depth_colormap))
        cv2.imshow("result", orig_im)
        out.write(orig_im)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        
        fps  = ( fps + (1./(time.time()-start)) ) / 2
        print("fps= %f"%(fps))


if __name__ =='__main__':
    main()

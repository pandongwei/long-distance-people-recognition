from mtcnn.src import detect_faces
from ArcFace.mobile_model import mobileFaceNet
from util_face_recognition import cosin_metric, get_feature, draw_ch_zn
import os
from torchvision import transforms
from PIL import Image,ImageFont
import time
from util_people_detection import *
from darknet import Darknet
from kalman_filter.tracker import Tracker
import json

#parameters from face recognition
font = ImageFont.truetype('simhei.ttf',20,encoding='utf-8')
cfgfile = "cfg/yolov3.cfg"
weightsfile ="model_data/yolov3.weights"
num_classes = 1
#parameters from people detection
classes = load_classes('data/coco.names')

#parameters for Kalman Filter


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


def write(x, img):
    c1 = tuple(x[1:3].astype(int))
    c2 = tuple(x[3:5].astype(int))
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2,(0,0,0), 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,(0,0,0), -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img

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

def main():
##########################################################################################################
    #preparation part

    with open('config/config.json', 'r') as f:
        cfg = json.load(f)

    confidence = float(0.25)
    nms_thesh = float(0.4)
    CUDA = torch.cuda.is_available()

    
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    
    model.net_info["height"] =  "160"
    inp_dim = int(model.net_info["height"])
    
    assert inp_dim % 32 == 0                   #assert后面语句为false时触发，中断程序
    assert inp_dim > 32
    
    if CUDA:
        model.cuda()
   
    model.eval()

    #Kalman Filter
    tracker = Tracker(dist_thresh = 160, max_frames_to_skip = 100, 
                                        max_trace_length = 5, trackIdCount = 1)


    saved_model = 'ArcFace/model/068.pth'
    name_list = os.listdir('users')
    path_list = [os.path.join('users', i, '%s.txt' % (i)) for i in name_list]
    total_features = np.empty((128,), np.float32)

    for i in path_list:
        temp = np.loadtxt(i)
        total_features = np.vstack((total_features, temp))
    total_features = total_features[1:]

    # threshold = 0.30896     #阈值并不合适，可能是因为训练集和测试集的差异所致！！！
    threshold = 0.5
    model_facenet = mobileFaceNet()
    model_facenet.load_state_dict(torch.load(saved_model)['backbone_net_list'])
    model_facenet.eval()
    use_cuda = torch.cuda.is_available() and True
    device = torch.device("cuda" if use_cuda else "cpu")
    # device = torch.device("cuda")

    # is_cuda_avilable
    trans = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    model_facenet.to(device)

    global person
    
    fps = 0.0
    count = 0
    frame = 0    
    person = []
    count_yolo = 0
    '''
    #record the video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output/test.avi',fourcc, 15.0, (640,480),True)
    '''
    cap = cv2.VideoCapture(0)
    
    detect_time = []
    recogn_time = []
    kalman_time = []
    aux_time = []
    while True:
        start = time.time()  
        ret, color_image = cap.read()
        if color_image is None:
            break

        img, orig_im, dim = prep_image(color_image, inp_dim)
        
        im_dim = torch.FloatTensor(dim).repeat(1,2)  
                
##################################################################################################
        #people detection part                
        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        time_a = time.time()
        if count_yolo %1 ==0:
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
            #output = sellect_person(output)                                       #把标签不是人的output去掉，减少计算量
            output = np.array(output)

            output_update = output
        elif count_yolo %1 != 0:
            output = output_update
        count_yolo += 1
        #list(map(lambda x: write(x, orig_im), output))                #把结果加到原来的图像中
        #output的[0,1:4]分别为框的左上和右下的点的位置
        detect_time.append(time.time() - time_a)
###########################################################################################################
        #kalman filter tracking part
        time_a = time.time()
        output_kalman_xywh = to_xy(output)                   #把output数据变成适合kalman更新的类型
        if (len(output_kalman_xywh) > 0):
            tracker.Update(output_kalman_xywh)                #用kalman filter更新框的位置
        
        outputs_kalman_normal = np.array(xy_to_normal(output,tracker.tracks)) #换回原来的数据形式
        kalman_time.append(time.time() - time_a)


        time_a = time.time()
        for output_kalman_normal in outputs_kalman_normal:
            #draw the bounding box
            left,top,right,down = [int(max(0,x)) for x in output_kalman_normal[0:4]]
            cv2.rectangle(orig_im, (left,top), (right,down),(255,255,255), 2)
            cv2.putText(orig_im, str(output_kalman_normal[4]),(left,top),0, 5e-3 * 200, (0,255,0),2)              #track id 就是数字

########################################################################################################
        #face recognition part
            person_img = orig_im[left:min(right,640),top:min(down,480)].copy()
            print(left,right,top,down)
            print(person_img.shape)
            img = Image.fromarray(person_img)
            bboxes, landmark = detect_faces(img)                                                                  #首先检测脸

            if len(bboxes) == 0:
                print('detect no people')
            else:
                print('the length  ', len(bboxes))
                for bbox in bboxes:
                    print(bbox)

                    loc_x_y = [bbox[2]+top, bbox[1]+left]
                    '''
                    cv2.imshow('person',person_img)
                    if cv2.waitKey(0) or 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                    '''
                    face_img = person_img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])].copy()              #从图像中截取框
                    feature = np.squeeze(get_feature(face_img, model_facenet, trans, device))                               #框里的图像计算feature
                    cos_distance = cosin_metric(total_features, feature)
                    index = np.argmax(cos_distance)
                    if  cos_distance[index] <= threshold:
                        continue
                    person = name_list[index]
                    #在这里加框加文字
                    orig_im = draw_ch_zn(orig_im,person,font,loc_x_y)                                                                    #加名字
                    cv2.rectangle(orig_im,(int(bbox[0]+top),int(bbox[1]+left)),(int(bbox[2]+top),int(bbox[3]+left)),(0,0,255))           #加box

        print('timetimetimetotal  ', time.time() - time_a)
###############################################################################################################
        time_a = time.time()
        '''
        #show the final output result
        for output_kalman_normal in outputs_kalman_normal:
            if output_kalman_normal[4] == 1:
                cv2.putText(orig_im, person, (output_kalman_normal[0].astype(np.int32)+100,output_kalman_normal[1].astype(np.int32)+20),
                            cv2.FONT_HERSHEY_PLAIN, 2, [0,255,0], 2)
        '''

        #out.write(orig_im)
        cv2.imshow("frame", orig_im)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        aux_time.append(time.time()-time_a)
        fps  = ( fps + (1./(time.time()-start)) ) / 2
        print("fps= %f"%(fps))
    
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

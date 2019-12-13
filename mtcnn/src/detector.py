import numpy as np
import torch
from torch.autograd import Variable
from .get_nets import PNet, RNet, ONet
from .box_utils import nms, calibrate_box, get_image_boxes, convert_to_square
from .first_stage import run_first_stage


def detect_faces(image, min_face_size=20.0,
                 thresholds=[0.6, 0.7, 0.8],
                 nms_thresholds=[0.7, 0.7, 0.7]):
    """
    Arguments:
        image: an instance of PIL.Image.
        min_face_size: a float number.
        thresholds: a list of length 3.                           #逐渐变大，从粗筛选到精筛选
        nms_thresholds: a list of length 3.

    Returns:
        two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
        bounding boxes and facial landmarks.                    #人脸框与人脸关键点
    """

    # LOAD MODELS
    pnet = PNet()
    rnet = RNet()
    onet = ONet()
    if torch.cuda.is_available() :
        pnet.cuda()
        rnet.cuda()
        onet.cuda()
    onet.eval()

    '''
    为了检测到不同size的人脸，在进入P-Net之前，我们应该对图像进行金字塔操作。
    首先，根据设定的min_face_size尺寸，将img按照一定的尺寸缩小，每次将img缩小到前级img面积的一半，形成scales列表，
    直至边长小于min_face_size，此时得到不同尺寸的输入图像。
    '''
    # BUILD AN IMAGE PYRAMID
    width, height = image.size
    min_length = min(height, width)

    min_detection_size = 12
    factor = 0.707  # sqrt(0.5),每次将img缩小到前级img面积的一半，就是边长的0.707

    # scales for scaling the image
    scales = []

    # scales the image so that
    # minimum size that we can detect equals to
    # minimum face size that we want to detect
    m = min_detection_size/min_face_size
    min_length *= m

    factor_count = 0
    while min_length > min_detection_size:
        scales.append(m*factor**factor_count)
        min_length *= factor
        factor_count += 1

    # STAGE 1

    '''
    Proposal Network (P-Net)：该网络结构主要获得了人脸区域的候选窗口和边界框的回归向量。
    并用该边界框做回归，对候选窗口进行校准，然后通过非极大值抑制（NMS）来合并高度重叠的候选框。
    '''
    # it will be returned
    bounding_boxes = []

    # run P-Net on different scales
    for s in scales:
        boxes = run_first_stage(image, pnet, scale=s, threshold=thresholds[0])
        bounding_boxes.append(boxes)
    
    #后面都是对box的后处理优化
    # collect boxes (and offsets, and scores) from different scales
    bounding_boxes = [i for i in bounding_boxes if i is not None]

    if bounding_boxes:   #add a condition
        bounding_boxes = np.vstack(bounding_boxes)                           #把数组给堆叠起来

        keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])              #Non-maximum suppression.
        bounding_boxes = bounding_boxes[keep]

        # use offsets predicted by pnet to transform bounding boxes
        bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])               #用reg*系列（对应坐标的线性回归参数）可对box进行坐标修正
        # shape [n_boxes, 5]

        bounding_boxes = convert_to_square(bounding_boxes)                     #目标框修正之后，先rec2square、再pad。rec2square是将修正后不规则的框调整为正方形，pad的目标是将超出原img范围的部分填充为0，大小比例不变。
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 2

        '''
        Refine Network (R-Net)：（跟P-Net结构作用差不多）该网络结构还是通过边界框回归和NMS来去掉那些false-positive区域。
        只是由于该网络结构和P-Net网络结构有差异，多了一个全连接层，所以会取得更好的抑制false-positive的作用。
        '''
        img_boxes = get_image_boxes(bounding_boxes, image, size=24)                   #将P-Net最后输出的所有box，resize到24*24后输入R-Net。
        img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
        if torch.cuda.is_available():
            img_boxes = img_boxes.cuda()
        output = rnet(img_boxes)
        offsets = output[0].data.numpy()  # shape [n_boxes, 4]
        probs = output[1].data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[1])[0]                               #继续筛选出大于阈值的boxes
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]

        keep = nms(bounding_boxes, nms_thresholds[1])
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])               #跟P-Net 类似的过程
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 3

        '''
        Output Network (O-Net)：该层比R-Net层又多了一层卷积层，所以处理的结果会更加精细。
        作用和R-Net层作用一样。但是该层对人脸区域进行了更多的监督，同时还会输出5个地标（landmark）。
        '''
        img_boxes = get_image_boxes(bounding_boxes, image, size=48)              #将R-Net最后输出的所有box，resize到48*48后输入O-Net。
        if len(img_boxes) == 0:
            return [], []
        img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
        if torch.cuda.is_available():
            img_boxes = img_boxes.cuda()
        output = onet(img_boxes)
        landmarks = output[0].data.numpy()  # shape [n_boxes, 10]            #比前面多了一个输出landmarks脸部的关键点位置，每张脸五个点
        offsets = output[1].data.numpy()  # shape [n_boxes, 4]
        probs = output[2].data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[2])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]
        landmarks = landmarks[keep]

        # compute landmark points
        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
        landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1)*landmarks[:, 0:5]
        landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1)*landmarks[:, 5:10]

        bounding_boxes = calibrate_box(bounding_boxes, offsets)
        keep = nms(bounding_boxes, nms_thresholds[2], mode='min')              #Non-maximum suppression.
        bounding_boxes = bounding_boxes[keep]
        landmarks = landmarks[keep]

        return bounding_boxes, landmarks
    else:
        return [],[]
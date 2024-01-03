import os
import cv2

from pycocotools.coco import COCO
import numpy as np

from os import path as op
from tqdm import tqdm

# colors = [[0,255,255], [135, 120, 28], [97, 68, 247], ]
colors = [[0,255,255],  [97, 68, 247], [110, 214, 179], [135, 120, 28], [53, 40, 186], [255, 228, 196], [82, 139, 139], [255, 160, 122]]
# colors = [[0,255,255],   [135, 120, 28], [53, 40, 186],]
# colors = [[0,255,255],  [53, 40, 186], [135, 120, 28], ]
# text_colors = [[255,255,255], [244, 106, 118], [35, 212, 183], [230, 236, 241]]
text_colors = [[255,255,255], [255,255,255], [255,255,255], [255,255,255], [255,255,255], [255,255,255], [255,255,255], [255,255,255]]




def NMS(dets, thresh):

    #x1、y1、x2、y2、以及score赋值
    # （x1、y1）（x2、y2）为box的左上和右下角标
    # import pdb;pdb.set_trace()
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]


    #每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #order是按照score降序排序的，得到的是排序的本来的索引，不是排完序的原数组
    order = scores.argsort()[::-1]
    # ::-1表示逆序


    temp = []
    while order.size > 0:
        i = order[0]
        temp.append(i)
        #计算当前概率最大矩形框与其他矩形框的相交框的坐标
        # 由于numpy的broadcast机制，得到的是向量
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.minimum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.maximum(y2[i], y2[order[1:]])

        #计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，需要用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #计算重叠度IoU
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        #找到重叠度不高于阈值的矩形框索引
        inds = np.where(ovr <= thresh)[0]
        #将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
    return temp


# DATASET='BCCD'
# json_file = r'DATASET/Microscopyimages/BCCD/annotations/test.json'
# pred_file = r'OUTPUT/bccd/bccd_zero/eval/glip_tiny_model_o365_goldg/inference/test/2022_09_02-05_53_33/bbox.json'
# dataset_dir = r'DATASET/Microscopyimages/BCCD/test/'
# output_dir = r'OUTPUT/show'

# tests = ['2022_09_27-11_09_54', '2022_09_27-06_44_57' , '2022_09_27-06_48_47',    '2022_09_27-06_50_04']
# tests = ['init', '+color', '+color+location', '+combain']
# subfix = ['init', '+color', '+color+location', '+combain']
# classnames = [[{1 : 'benign', 2 : 'malignant'}], [{1 : 'brown mark', 2 : 'dark spot'}], [{1 : 'round mark at skin', 2 : 'uneven border  spot at skin'}],  [{1 : 'brown round mark at skin', 2 : 'uneven border dark spot at skin'}]]
# classnames = [[{1 : 'polyp', }], [{1 : 'pink bump', }], [{1 : 'pink bump in rectum',}],  [{1 : 'pink oval bump in rectum', }]]
# classnames = [[{1 : 'wound', }],]
# classnames = [[{ 1: "No abnormal"}], [{2: "Flare"}], [{3: "Purulent"}], [{4: "Scab"}], [{5: "Tension blisters"}], [{6: "Disruption of wound"}], [{7: "Skin ecchymosis around incision"}]]
# # tests = [ '2022_09_27-06_48_47', ]


# json_file = r'/DATA/BUV2022/annotations/test.json' #'/DATA/POLYP/annotations/CVC-300_test.json'
json_file =r'/DATA/CHAOS/MRI/T1-DUAL/annotations/test_t1.json' #r'/DATA/CHAOS/CT/annotations/test.json'
# dataset_dir = r'/project/project/GLIP/MIU-VL/DATA/BUV2022/images/' #'/project/project/GLIP/MIU-VL/DATA/POLYP/test/CVC-300/images/'
dataset_dir = r'/project/project/GLIP/MIU-VL/DATA/CHAOS/MRI/T1-DUAL/T1LiverImages/'#r'/project/project/GLIP/MIU-VL/DATA/CHAOS/CT/LiverImages/'  
# DATASET='isbi2016/{}'.format(sub)
# prefix = "OUTPUT/isbi2016/isbi2016_zero/eval/glip_tiny_model_o365_goldg/inference/test/"
# json_file = r'DATASET/Naturalimages/ISBI2016/annotations/test.json'
# dataset_dir = r'DATASET/Naturalimages/ISBI2016/images/test/'


# pred_file = r"/OUTPUTS/polyp/hybrid/zero_shot/buv/llm/top1/eval/glip_tiny_model_o365_goldg/inference/val/bbox.json"#r"/OUTPUTS/polyp/hybrid/zero_shot/cvc300/llm/eval/glip_tiny_model_o365_goldg/inference/val/bbox.json"
pred_file = r'/OUTPUTS/t1/zero_shot/llm/top2/eval/glip_tiny_model_o365_goldg/inference/val/bbox.json' #r'/OUTPUTS/ct_liver/zero_shot/llm/top1/eval/glip_tiny_model_o365_goldg/inference/val/bbox.json'
method=r"nms"


output_dir = r'./show_wound/mri_t1/'

show_thr = 0.1
cur_path = os.getcwd()
# import pdb; pdb.set_trace()
coco = COCO(cur_path + json_file)
results = coco.loadRes(cur_path + pred_file)
imgIds = coco.getImgIds()  # 图片id，许多值

for imgId in tqdm(imgIds):
    img = coco.loadImgs(imgId)[0]
    image = cv2.imread(dataset_dir + img['file_name'])
    lw = max(round(sum(image.shape) / 2 * 0.004), 2)  # line width
    color_seg = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    # import pdb;pdb.set_trace()
    pred_ids = results.getAnnIds(imgId)
    preds = []
    for pred_id in pred_ids:
        pred = results.loadAnns(pred_id)[0]
        
        pred_bbox = pred['bbox']
        p_xmin = int(pred_bbox[0])
        p_ymin = int(pred_bbox[1])
        p_xmax = int(pred_bbox[2]) + p_xmin
        p_ymax = int(pred_bbox[3]) + p_ymin
        pred_score = pred['score']
        preds.append([p_xmin, p_ymin, p_xmax, p_ymax, pred_score, pred['category_id']] )
    # import pdb;pdb.set_trace()
    if len(preds)  > 0:
        preds = np.array(preds)
        
        nms_pids = NMS(preds, 0.3) #threshold
        for pid in nms_pids:
            p_xmin = int(preds[pid][0])
            p_ymin = int(preds[pid][1])
            p_xmax = int(preds[pid][2])
            p_ymax = int(preds[pid][3])
            pred_score = preds[pid][4]
            cid = int(preds[pid][5])
            # if cid != 1:
            #     import pdb;pdb.set_trace()
            cate_name = coco.cats[cid]['name']

            if pred_score > show_thr:
                color = colors[cid]
                # color_seg[p_xmin:p_xmax, p_ymin:p_ymax, :] = color
                # image = image*(1-0.2) + color_seg * 0.2
                text_color = text_colors[0]
                text = cate_name + " " + str('{:.2%}'.format(pred_score))
                p1, p2 = (int(p_xmin), int(p_ymin)), (int(p_xmax), int(p_ymax)) #bbox generate
                image = cv2.rectangle(image, p1, p2, color=color, thickness=lw, lineType=cv2.LINE_AA)
                # image = cv2.putText(image, text ,(p_xmin + 3, p_ymin - 3), 
                #             cv2.FONT_HERSHEY_COMPLEX, .3, text_color, 1)
                tf = max(lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(text, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(image,
                            text, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                            0,
                            lw / 3,
                            text_color,
                            thickness=tf,
                            lineType=cv2.LINE_AA)

                out_img = op.join(output_dir, method, img['file_name'])
                if not op.exists(op.dirname(out_img)):
                    os.makedirs(op.dirname(out_img))
                # import pdb;pdb.set_trace()
                cv2.imwrite(out_img, image)




### Different path
YOLO_master = "C:\\Users\\t1526\\Desktop\\kaggle\\input\\ultralytics-main\\"
CRNN_master = "C:\\Users\\t1526\\Desktop\\kaggle\\input\\crnn-pytorch-master\\"
PATH_images = "C:\\Users\\t1526\\Desktop\\kaggle\\input\\benetech-making-graphs-accessible\\train\images\\"
Model_path_ODT = "C:\\Users\\t1526\\Desktop\\kaggle\\input\\ultralytics-main\\modelODT.pt"
Model_path_BHR = "C:\\Users\\t1526\\Desktop\\kaggle\\input\\ultralytics-main\\bar_height_recognise.pt"
Model_path_LHR = "C:\\Users\\t1526\\Desktop\\kaggle\\input\\ultralytics-main\\line_height_recognise.pt"
Model_path_IWR = "C:\\Users\\t1526\\Desktop\\kaggle\\input\\crnn-pytorch-master\\netCRNN.model"
Model_path_STR = "C:\\Users\\t1526\\Desktop\\kaggle\\input\\ultralytics-main\\scatter_recognise.pt"

### Object type
# CHART_TYPE = {'vertical_bar':0,'horizontal_bar':1,'dot':2,'line':3,'scatter':4}
# OBJ_TYPE = {'label_angle00':5,'label_angle45':6,'label_angle90':7,'label_angle-45':8,
#             'axis_x':9,'axis_y':10,'tick_point_x':11,'tick_point_y':12,'useless':13}
OBJ_TYPE = ['vertical_bar','horizontal_bar','dot','line','scatter',
#                 0               1           2      3       4
            'label_angle00','label_angle45','label_angle90','label_angle-45','useless']
#                 5               6                 7               8            9

### Other const varies
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sys
sys.path.append(YOLO_master)
sys.path.append(CRNN_master)

### pandas result saving
import pandas as pd
result_df = pd.DataFrame(columns=['id', 'data_series', 'chart_type'])





### Prepare dataset

from torchvision.transforms import PILToTensor
from PIL import Image
import torch, os

class DataLoader:
    def __init__(self, path_images):
        _,_,self.image_list = next(os.walk(path_images))
        self.data_length = len(self.image_list)
        print("Total number of test pictures:", self.data_length)
        self.image_list.sort()
        self.iter_count = -1
    def __len__(self):
        return self.data_length
    def __getitem__(self, item):
        img = Image.open(PATH_images+self.image_list[item])
        return img, self.image_list[item]
    def __iter__(self):
        self.iter_count = -1
        return self
    def __next__(self):
        self.iter_count += 1
        if self.iter_count == self.data_length:
            raise StopIteration
        return self.__getitem__(self.iter_count)

dataloader = DataLoader(PATH_images)
# dataloader: [for img,information in dataloader:]





# load object detect model (ODT)
from ultralytics import YOLO
modelODT = YOLO(Model_path_ODT)
def ODT(img):
    result = modelODT.predict(source=img)
    return result[0].boxes.data
# function: ODT: input:img; output:result


# load bar height recognise model (BHR)
from ultralytics import YOLO
modelBHR = YOLO(Model_path_BHR)
def BHR(img):
    result = modelBHR.predict(source=img)
    return result[0].boxes.data
# function: BHR: input:img; output:result


# load line height recognise model (LHR)
from ultralytics import YOLO
modelLHR = YOLO(Model_path_LHR)
def LHR(img):
    result = modelLHR.predict(source=img)
    return result[0].boxes.data
# function: LHR: input:img; output:result


# load scatter recognise model (LHR)
from ultralytics import YOLO
modelSTR = YOLO(Model_path_STR)
def STR(img):
    result = modelSTR.predict(source=img)
    return result[0].boxes.data
# function: STR: input:img; output:result


# load dot number recognise model(manual) (DNR)
import cv2
import numpy as np
def DNR(img):
    count = 0
    img = img.convert("L")
    img = np.asarray(img)
    img = img[:-5][:] # To avoid edge problem
    img = cv2.bilateralFilter(img,9,50,30)
    if img.std()<10: return 0 # Nothing in the picture
    img = (img-img.mean())/img.std()
    feature = img.mean(axis=1)
    if img.std()<0.02: return 0 # Nothing in the picture
    bound = (np.max(feature)+np.min(feature))/2
    if feature[0]<bound: count+=1 # All filled with dots
    for i in range(len(feature)-1):
        if (feature[i]-bound)*(feature[i+1]-bound)<=0:
            count+=1 # An intersection point
    return (count+1)//2
# function: DNR: input:img; output:result





# load image word reader model (IWR)

import torch
import utils
import params
from PIL import Image
from models.crnn import CRNN
from torch.autograd import Variable
import torchvision.transforms as transforms

# init
class resizeNormalize:
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img
nclass = len(params.alphabet) + 1
model = torch.load(Model_path_IWR).to(DEVICE)
model.eval()
converter = utils.strLabelConverter(params.alphabet)
transformer = resizeNormalize((400, 64))

def IWR(image):
    image = image.convert('L')
    image = transformer(image)
    image = image.to(DEVICE)
    image = image.view(1, *image.size())
    image = Variable(image)
    preds = model(image)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.LongTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    return sim_pred

# function: IWR: input:img; output:words





### main

def doODT(img):
    def get_label_classify_function(chart_box):
        x1 = chart_box[0]
        x2 = (chart_box[2]-chart_box[0])/3+chart_box[0]
        y1 = chart_box[3]
        y2 = chart_box[1]
        k = (y2-y1)/(x2-x1)
        b = y1-k*x1
        def label_classify_function(label_box):
            x = (label_box[0]+label_box[2])/2
            y = (label_box[1]+label_box[3])/2
            if k*x+b>y and x<chart_box[0]+10: 
                return "y"
            elif k*x+b<=y and x<=chart_box[2]+10 and y>=chart_box[3]-2: 
                return "x"
        return label_classify_function
    labels_x = []
    labels_y = []
    chart_type = None
    chart_box = None
    ### object detect (ODT)
    resultODT = ODT(img)
    boxes = list(resultODT.cpu().numpy())
    # get main chart position
    for obj in boxes:
        object_class_number = int(obj[-1]+0.5)
        if 0<=object_class_number<=4: # chart_type
            chart_type = OBJ_TYPE[object_class_number]; 
            chart_box = list(map(lambda x:int(x+0.5), obj[0:4]))
            break # need not go on
    # get label classify function
    if chart_type is None:
        print(f"Error: main chart is not found in {information}.\n\tAssert type is scatter and continue...")
        chart_type = 'scatter'
    label_classify_function = get_label_classify_function(chart_box)
    # get labels
    for obj in boxes:
        object_class_number = int(obj[-1]+0.5)
        if 5<=object_class_number<=8:
            label_type = OBJ_TYPE[object_class_number]
            label_box = list(map(lambda x:int(x+0.5), obj[:4]))
            if label_type!="label_angle00":
                labels_x.append([label_type, label_box])
            elif label_classify_function(label_box)=="y":
                labels_y.append([label_type, label_box])
            else: # label_classify_function(label_box)=="x":
                labels_x.append([label_type, label_box])
    # chart_type, chart_box, labels_x, labels_y
    return chart_type, chart_box, labels_x, labels_y





def check_and_solve_label_x(labels_x):
    def iou_filiter(labels_x):
        def IoU(box1, box2): # Intersection over Union
            x1,y1,x2,y2 = box1
            a1,b1,a2,b2 = box2
            ax=max(x1,a1); ay=max(y1,b1) 
            bx=min(x2,a2); by=min(y2,b2)
            area_N = (x2-x1)*(y2-y1)
            area_M = (a2-a1)*(b2-b1)
            w=bx-ax; h=by-ay
            if w<=0 or h<=0: 
                return 0, area_N, area_M
            area_X = w*h
            iou = area_X/(area_N+area_M-area_X)
            return iou, area_N, area_M
        for i in reversed(range(len(labels_x)-1)):
            if (labels_x[i+1][1][0]-labels_x[i][1][0])*(labels_x[i+1][1][2]-labels_x[i][1][2])<=0:
                _, area1, area2 = IoU(labels_x[i][1], labels_x[i+1][1])
                if area1>area2: del labels_x[i+1]
                else: del labels_x[i]
                continue
            iou, area1, area2 = IoU(labels_x[i][1], labels_x[i+1][1])
            if iou<0.3: pass # iou test passed
            else: # iou test not passed
                if area1>area2: del labels_x[i+1]
                else: del labels_x[i]
        return labels_x
    def repair_fixer(labels_x):
        index = distances.index(max(distances))
        # loss one between labels_x[index],labels_x[index+1]
        new_box_0 = labels_x[index][1][2]
        new_box_1 = max(labels_x[index][1][1],labels_x[index+1][1][1])
        new_box_2 = labels_x[index+1][1][0]
        new_box_3 = min(labels_x[index][1][3],labels_x[index+1][1][3])
        new_box = (new_box_0,new_box_1,new_box_2,new_box_3)
        new_type = labels_x[index][0]
        labels_x.append([new_type,new_box])
        labels_x.sort(key=lambda x:(x[1][0]+x[1][2])/2)
        return labels_x
    def too_close_fixer(labels_x):
        index = distances.index(min(distances))
        # labels_x[index] & labels_x[index+1] is too close
        length1 = (labels_x[index][1][2]-labels_x[index][1][0])
        length2 = (labels_x[index+1][1][2]-labels_x[index+1][1][0])
        if length1>=length2: del labels_x[index+1]
        else: del labels_x[index]
        return labels_x
    def too_close_fixer_45(labels_x):
        index = distances.index(min(distances))
        # labels_x[index] & labels_x[index+1] is too close
        length1 = (labels_x[index][1][2]-labels_x[index][1][0])
        length2 = (labels_x[index+1][1][2]-labels_x[index+1][1][0])
        if length1<=length2: del labels_x[index+1]
        else: del labels_x[index]
        return labels_x
    def IWR_duplication_filiter_45(labels_x):
        words_list = []
        length = len(labels_x)
        for i,label in enumerate(reversed(labels_x)):
            i = length-i-1
            crop_box = (label[1][0]-3,label[1][1]-3,label[1][2]+3,label[1][3]+3)
            cropped_img = img.crop(crop_box)
            cropped_img = cropped_img.resize((cropped_img.size[0]*8, cropped_img.size[1]*8))
            cropped_img = cropped_img.rotate(-45, expand=True)
            crop_box = (60, int(cropped_img.size[1]/2-70+0.5),
                        cropped_img.size[0]-60, int(cropped_img.size[1]/2+70+0.5))
            cropped_img = cropped_img.crop(crop_box)
            words = IWR(cropped_img)
            if words in words_list:
                del labels_x[i]
            else: # words not in words_list
                words_list.append(words)
        return labels_x
    def loss_fixer(labels_x):
        labels_x_ori = labels_x.copy()
        distance_min = min(distances)
        centers = list(map(lambda x:x[1][2], labels_x))
        ptr = chart_box[0]
        if (centers[0]-ptr)<distance_min: 
            ptr = centers[0]
        else: # (centers[0]-ptr)>=distance_min: 
            blocks = (centers[0]-ptr)//distance_min
            ptr = centers[0]-blocks*(distance_min+0.5)
        max_height = max(labels_x, key=lambda x:x[1][3])[1][3]
        min_height = min(labels_x, key=lambda x:x[1][1])[1][1]
        # print(max_height,min_height)
        count = 0
        while count<50: # break in circle
            count += 1
            for _,box in labels_x:
                if box[2]-15<=ptr<=box[2]+15: 
                    ptr = box[2] + distance_min+0.5; break
            else: # no label box
                box = [int(ptr-(max_height-min_height)+1.7),min_height,
                       int(ptr+1.7),max_height]
                labels_x.append(('label_angle45',box))
                ptr += (distance_min+0.5)
            if ptr>chart_box[2]: break
        else: # dead circle
            labels_x = labels_x_ori
            print(f"\tGot out of a dead circle in {information}")
        labels_x.sort(key=lambda x:x[1][2])
        return labels_x
# functions above
    # label type check and solve
    label_type_dict = {}
    for label in labels_x:
        if label[0] in label_type_dict.keys():
            label_type_dict[label[0]] += 1
        else: # label[0] not in label_type_dict.key():
            label_type_dict[label[0]] = 1
    label_type = max(label_type_dict, key=lambda x:label_type_dict[x])
    new_labels_x = []
    for label in labels_x:
        if label[0]==label_type:
            new_labels_x.append(label)
        else: # label classify error
            print(f"Warning: label type different in x-axis in {information}:\n",
                  f"\t{label} appeared in {label_type}")
    labels_x = new_labels_x
# label_type=="label_angle00" or label_type=="label_angle90"
    if label_type=="label_angle00" or label_type=="label_angle90":
        labels_x.sort(key=lambda x:(x[1][0]+x[1][2])/2)
        # checking
        distances = [(labels_x[i+1][1][0]+labels_x[i+1][1][2])/2
                     -(labels_x[i][1][0]+labels_x[i][1][2])/2 for i in range(len(labels_x)-1)]
        if max(distances)-min(distances) < 12: 
            return labels_x
        print(f"Warning: Suspicious distances in x-axis in {information}; trying to fix...")
        # iou filiter
        labels_x = iou_filiter(labels_x)
        distances = [(labels_x[i+1][1][0]+labels_x[i+1][1][2])/2
                    -(labels_x[i][1][0]+labels_x[i][1][2])/2 for i in range(len(labels_x)-1)]
        if max(distances)-min(distances) < 12:
            print(f"\tIoU fix successed in {information}")
            return labels_x
        # loss repair
        if min(distances)-10 < max(distances)/2 < min(distances)+10:
            labels_x = repair_fixer(labels_x)
        distances = [(labels_x[i+1][1][0]+labels_x[i+1][1][2])/2
                    -(labels_x[i][1][0]+labels_x[i][1][2])/2 for i in range(len(labels_x)-1)]
        if max(distances)-min(distances) < 12:
            print(f"\tRepair fix successed in {information}")
            return labels_x
        # too close fix
        if min(distances)<1: # close enough
            labels_x = too_close_fixer(labels_x)
        distances = [(labels_x[i+1][1][0]+labels_x[i+1][1][2])/2
                    -(labels_x[i][1][0]+labels_x[i][1][2])/2 for i in range(len(labels_x)-1)]
        if max(distances)-min(distances) < 12:
            print(f"\tToo close fix successed in {information}")
            return labels_x
        # still something wrong
        print(f"Error: All fix failed in {information}; Just continue...  \n\tdistances:{distances}")
        return labels_x
# label_type=="label_angle45"
    if label_type=="label_angle45":
        labels_x.sort(key=lambda x:x[1][2])
        # checking
        distances = [labels_x[i+1][1][2]-labels_x[i][1][2] for i in range(len(labels_x)-1)]
        if max(distances)-min(distances)<18:
            return labels_x # no problem
        print(f"Warning: Suspicious distances in x-axis 45 in {information}; trying to fix...")
        # IWR duplication filiter
        labels_x = IWR_duplication_filiter_45(labels_x)
        distances = [labels_x[i+1][1][2]-labels_x[i][1][2] for i in range(len(labels_x)-1)]
        if max(distances)-min(distances)<18:
            print(f"\tIWR deduplication fix 45 successed in {information}",len(labels_x))
            return labels_x
        # too close fix
        if min(distances)<=7:
            labels_x = too_close_fixer_45(labels_x)
        distances = [labels_x[i+1][1][2]-labels_x[i][1][2] for i in range(len(labels_x)-1)]
        if max(distances)-min(distances)<18:
            print(f"\tToo close fix 45 successed in {information}",len(labels_x))
            return labels_x
        # ignore right numbers ############################################################# can be updated
        if len(distances)==9 or len(distances)==14:
            print(f"\tIgnore: Ignore this problem in {information}. length may be right",len(labels_x))
            return labels_x
        # fix loss
        if len(distances)<=14:
            labels_x = loss_fixer(labels_x)
        distances = [labels_x[i+1][1][2]-labels_x[i][1][2] for i in range(len(labels_x)-1)]
        if max(distances)-min(distances)<18:
            print(f"\tLoss fix 45 successed in {information}",len(labels_x))
            return labels_x
        # still something wrong
        else: # still something wrong
            if len(labels_x)>15:
                labels_x = labels_x[:15]
            else: # len(labels_x)<=14
                lost_numbers = 9-len(distances) if len(distances)<=9 else 14-len(distances)
                for _ in range(lost_numbers): labels_x.append(labels_x[-1])
            print(f"Ignore: Fix 45 number and Ignore this problem in {information}",len(labels_x))
            return labels_x
        # still something wrong
        print(f"Error: All fix 45 failed in {information}; Just continue...  \n\tdistances:{distances}")
        return labels_x
# label_type=="label_angle-45"
    if label_type=="label_angle-45":
        labels_x.sort(key=lambda x:x[1][0])
        distances = [labels_x[i+1][1][1]-labels_x[i][1][1] for i in range(len(labels_x)-1)]
        if max(distances)-min(distances)<18:
            return labels_x
        else: # abnormal
            print(f"Error: Distances -45 error in {information}.")
# any label else
    print(f"Error: Unknown label type {label_type} in {information}; Just continue...")
    return labels_x

def check_and_solve_label_y(labels_y):
    def iou_filiter(labels_x):
        def IoU(box1, box2): # Intersection over Union
            x1,y1,x2,y2 = box1
            a1,b1,a2,b2 = box2
            ax=max(x1,a1); ay=max(y1,b1) 
            bx=min(x2,a2); by=min(y2,b2)
            area_N = (x2-x1)*(y2-y1)
            area_M = (a2-a1)*(b2-b1)
            w=bx-ax; h=by-ay
            if w<=0 or h<=0: 
                return 0, area_N, area_M
            area_X = w*h
            iou = area_X/(area_N+area_M-area_X)
            return iou, area_N, area_M
        for i in reversed(range(len(labels_x)-1)):
            iou, area1, area2 = IoU(labels_x[i][1], labels_x[i+1][1])
            if iou<0.3: pass # iou test passed
            else: # iou test not passed
                if area1>area2: del labels_x[i+1]
                else: del labels_x[i]
        return labels_x
# functions above
    # label type check and solve
    label_type_dict = {}
    for label in labels_y:
        if label[0] in label_type_dict.keys():
            label_type_dict[label[0]] += 1
        else: # label[0] not in label_type_dict.key():
            label_type_dict[label[0]] = 1
    label_type = max(label_type_dict, key=lambda y:label_type_dict[y])
    new_labels_y = []
    for label in labels_y:
        if label[0]==label_type:
            new_labels_y.append(label)
        else: # label classify error
            print(f"Warning: label type different in x-axis in {information}:\n",
                  f"\t{label} appeared in {label_type}")
    labels_y = new_labels_y
# label_type=="label_angle00" or label_type=="label_angle90"
    if label_type=="label_angle00":
        labels_y.sort(key=lambda y:(y[1][1]+y[1][3])/2)
        # checking
        distances = [(labels_y[i+1][1][1]+labels_y[i+1][1][3])/2
                     -(labels_y[i][1][1]+labels_y[i][1][3])/2 for i in range(len(labels_y)-1)]
        if max(distances)-min(distances) < 10: 
            return labels_y
        print(f"Warning: Suspicious distances in y-axis in {information}; trying to fix...")
        # iou filiter
        labels_y = iou_filiter(labels_y)
        distances = [(labels_y[i+1][1][1]+labels_y[i+1][1][3])/2
                     -(labels_y[i][1][1]+labels_y[i][1][3])/2 for i in range(len(labels_y)-1)]
        if max(distances)-min(distances) < 10: 
            print(f"\tIoU fix successed in {information}")
            return labels_y
        else: # failed
            print(f"Error: All fix failed in {information}; Just continue...  \n\tdistances:{distances}")
            return labels_y
# any label else
    print(f"Error: Unknown label type {label_type} in {information}; Just continue...")
    return labels_x





def smart_float(words):
    if words.count(".")>=2: words = words.replace(".", '');
    words = words.replace(",", '');
    words = words.replace(" ", '');
    words = words.replace("O", '0');
    words = words.replace("o", '0');        
    try: # avoid float changing problem
        if words[-1]=='%': words = float(words[:-1])
        elif words[0]=='$': words = float(words[1:])
        else: words = float(words)
        return words
    except: # avoid float changing problem
        return words
def get_transform_function(label_ticks_y, axis='y'):
    def choose_great_label(label):
        crop_box = (label[1][0]-3, label[1][1]-2, label[1][2]+3, label[1][3]+2)
        cropped_img = img.crop(crop_box)
        words = IWR(cropped_img)
        return smart_float(words)
    if axis=="y": label_ticks_y.sort(key=lambda y:(y[1][1]+y[1][3])/2)
    else: label_ticks_y.sort(key=lambda y:(y[1][0]+y[1][2])/2)
# pos1
    number = "False"
    if len(label_ticks_y)<=4: times_count = 0
    else: times_count = 1 # To avoid 0 which is not accurate
    while times_count<len(label_ticks_y):
        number = choose_great_label(label_ticks_y[times_count])
        if isinstance(number,float): break
        else: times_count+=1
    else: # give up
        print(f"Error: Dead circle in transform_y in {information}; And continue...\n\tWords:{number}")
        return lambda x:-x
    index = times_count
    if axis=="y": picture_pos = (label_ticks_y[index][1][1]+label_ticks_y[index][1][3])/2
    else: picture_pos = (label_ticks_y[index][1][0]+label_ticks_y[index][1][2])/2
    point1 = {"x":picture_pos, "y":number}
# pos2
    number = "False"
    if len(label_ticks_y)<=4: times_count = 1
    else: times_count = 2 # To avoid 0 which is not accurate
    while times_count<len(label_ticks_y):
        number = choose_great_label(label_ticks_y[len(label_ticks_y)-times_count])
        if isinstance(number,float): break
        times_count+=1
    else: # give up
        print(f"Error: Dead circle in transform_y in {information}; And continue...\n\tWords:{number}")
        return lambda x:-x
    index = len(label_ticks_y)-times_count
    if axis=="y": picture_pos = (label_ticks_y[index][1][1]+label_ticks_y[index][1][3])/2
    else: picture_pos = (label_ticks_y[index][1][0]+label_ticks_y[index][1][2])/2
    point2 = {"x":picture_pos, "y":number}
# check
    if abs(point1['x']-point2['x'])+abs(point1['y']-point2['y'])<0.001:
        print(f"Error: Not enough information to ensure axis in {information}")
        return lambda x:-x
    # get function
    k = (point2['y']-point1['y'])/(point2['x']-point1['x'])
    b = point1['y']-point1['x']*k   
    return lambda x:k*x+b






def doIWR(label):
    crop_box = (label[1][0]-3,label[1][1]-3,label[1][2]+3,label[1][3]+3)
    cropped_img = img.crop(crop_box)
    cropped_img = cropped_img.resize((cropped_img.size[0]*8, cropped_img.size[1]*8))
    if label[0]=='label_angle00':
        pass # need not to process
    elif label[0]=='label_angle90':
        cropped_img = cropped_img.rotate(-90)
    elif label[0]=='label_angle45':
        cropped_img = cropped_img.rotate(-45, expand=True)
        crop_box = (60, int(cropped_img.size[1]/2-70+0.5),
                   cropped_img.size[0]-60, int(cropped_img.size[1]/2+70+0.5))
        cropped_img = cropped_img.crop(crop_box)
    elif label[0]=='label_angle-45':
        cropped_img = cropped_img.rotate(45, expand=True)
        crop_box = (60, int(cropped_img.size[1]/2-70+0.5),
                    cropped_img.size[0]-60, int(cropped_img.size[1]/2+70+0.5))
        cropped_img = cropped_img.crop(crop_box)
    words = IWR(cropped_img)
    return words

def doBHR(label):
    if label[0]=='label_angle00' or label[0]=='label_angle90':
        center = (label[1][0]+label[1][2])/2
    elif label[0]=='label_angle45':
        center = label[1][2]-5
    elif label[0]=='label_angle-45':
        center = label[1][0]+5
    half_width = (chart_box[3]-(chart_box[1]-10))/8/2
    crop_box = (int(center-half_width+0.5), chart_box[1]-10, int(center+half_width+0.5), chart_box[3])
    cropped_img = img.crop(crop_box)
    # cropped_img.show()
    position = BHR(cropped_img)
    if len(position)==0: 
        true_height=chart_box[3]-0.8
    else: # recognised
        position = list(position.cpu().numpy())
        min_index = None
        nearest = 100
        for i,obj in enumerate(position):
            _temp = abs((obj[1]+obj[3])/2/cropped_img.size[1]-0.5)
            if _temp<nearest:
                _temp = nearest
                min_index = i
        height = (position[i][1]+position[i][3])/2
        true_height = height+chart_box[1]-10
    return true_height

def doBHR_horizontal(label):
    center = (label[1][1]+label[1][3])/2
    half_width = (chart_box[2]-chart_box[0])/8/2
    crop_box = (chart_box[0], int(center-half_width+0.5), chart_box[2], int(center+half_width+0.5))
    cropped_img = img.crop(crop_box).transpose(Image.ROTATE_90)
    cropped_img.show()
    position = BHR(cropped_img)
    if len(position)==0: 
        true_height=chart_box[0]+0.8
    else: # recognised
        position = list(position.cpu().numpy())
        min_index = None
        nearest = 100
        for i,obj in enumerate(position):
            _temp = abs((obj[1]+obj[3])/2/cropped_img.size[1]-0.5)
            if _temp<nearest:
                _temp = nearest
                min_index = i
        height = (position[i][1]+position[i][3])/2
        true_height = cropped_img.size[1]-height+chart_box[0]
    return true_height

def doLHR(label, position='middle'):
    if label[0]=='label_angle00' or label[0]=='label_angle90':
        center = (label[1][0]+label[1][2])/2
    elif label[0]=='label_angle45':
        center = label[1][2]-5
    elif label[0]=='label_angle-45':
        center = label[1][0]+5
    half_width = (chart_box[3]-(chart_box[1]-10))/8/2
    if position=="middle":
        crop_box = (int(center-half_width+0.5), chart_box[1]-10, int(center+half_width+0.5), chart_box[3])
    elif position=="left":
        crop_box = (int(center+0.5), chart_box[1]-10, int(center+half_width*2+0.5), chart_box[3])
    elif position=="right":
        crop_box = (int(center-half_width*2+0.5), chart_box[1]-10, int(center+0.5), chart_box[3])
    cropped_img = img.crop(crop_box)
    # cropped_img.show()
    position = LHR(cropped_img)
    if len(position)==0: 
        true_height=chart_box[3]-0.8
    else: # recognised
        position = list(position.cpu().numpy())
        min_index = None
        nearest = 100
        for i,obj in enumerate(position):
            _temp = abs((obj[1]+obj[3])/2/cropped_img.size[1]-0.5)
            if _temp<nearest:
                _temp = nearest
                min_index = i
        height = (position[i][1]+position[i][3])/2
        true_height = height+chart_box[1]-10
    return true_height

def doDNR(label):
    if label[0]=='label_angle00' or label[0]=='label_angle90':
        center = (label[1][0]+label[1][2])/2
    elif label[0]=='label_angle45':
        center = label[1][2]-5
    elif label[0]=='label_angle-45':
        center = label[1][0]+5
    half_width = (chart_box[3]-(chart_box[1]-10))/8/2
    crop_box = (int(center-half_width+0.5)+3, chart_box[1]-10, int(center+half_width+0.5)-3, chart_box[3])
    cropped_img = img.crop(crop_box)
    # cropped_img.show()
    number = DNR(cropped_img)
    return number

def doSTR(img, transform_function_x, transform_function_y):
    scatters = STR(img)
    processed_scatters = []
    for scatter in scatters.cpu().numpy():
        x = (scatter[0]+scatter[2])/2
        y = (scatter[1]+scatter[3])/2
        x = transform_function_x(x)
        y = transform_function_y(y)
        processed_scatters.append([x,y])
    processed_scatters.sort(key=lambda x:x[0])
    return processed_scatters

def save_to_df(result_list):
    global result_df
    global result_df_writed
    # x-series
    value2 = ""
    for i in result_list: value2 += (str(i[0])+";")
    value2 = value2[:-1]
    result_df.loc[len(result_df.index)] = [information[:-4]+"_x", value2, chart_type]
    # y-series
    value2 = ""
    for i in result_list: value2 += (str(i[1])+";")
    value2 = value2[:-1]
    result_df.loc[len(result_df.index)] = [information[:-4]+"_y", value2, chart_type]
    # mark
    result_df_writed = True

    
    
    
    
    

for i,(img, information) in enumerate(dataloader):
    result_df_writed = False
    
    try: 
        
    ### object detect (ODT)
        chart_type, chart_box, labels_x, labels_y = doODT(img)
        if chart_type!='horizontal_bar':
            labels_x = check_and_solve_label_x(labels_x)
        else: # chart_type=='horizontal_bar'
            labels_y = check_and_solve_label_y(labels_y)
        result_list = []
        
    ### classes process
        if chart_type=='vertical_bar':
            transform_function = get_transform_function(labels_y, axis="y")
            result_list = []
            for label in labels_x:
                words = doIWR(label)
                true_height = doBHR(label)
                number = transform_function(true_height)
                result_list.append([words,number])
            save_to_df(result_list)
        elif chart_type=='horizontal_bar':
            transform_function = get_transform_function(labels_x, axis="x")
            result_list = []
            for label in labels_y:
                words = doIWR(label)
                true_height = doBHR_horizontal(label)
                number = transform_function(true_height)
                result_list.append([number,words])
            save_to_df(result_list)
        elif chart_type=='line':
            transform_function = get_transform_function(labels_y, axis="y")
            result_list = []
            index = 0
            while index<len(labels_x):
                words = doIWR(labels_x[index])
                if index==0: true_height = doLHR(labels_x[index], position="left")
                elif index==len(labels_x)-1: true_height = doLHR(labels_x[index], position="right")
                else: true_height = doLHR(labels_x[index], position="middle")
                number = transform_function(true_height)
                result_list.append([words,number])
                index += 1
            save_to_df(result_list)
        elif chart_type=='dot':
            result_list = []
            for label in labels_x:
                words = doIWR(label)
                number = doDNR(label)
                result_list.append([words,number])
            save_to_df(result_list)
        elif chart_type=='scatter':
            transform_function_y = get_transform_function(labels_y, axis='y')
            transform_function_x = get_transform_function(labels_x, axis='x')
            processed_scatters = doSTR(img, transform_function_x, transform_function_y)
            save_to_df(processed_scatters)
            
    ### Unexpected
    except Exception as error:
        print(f"Error: Unexpected error: {error} in {information}")
    finally: # avoid block
        if result_df_writed==False:
            save_to_df([["unknown","unknown"]])
            print(f"Inform: Write unknown to {information}")
    
    
    
    
    
### save
result_df.to_csv("./submission.csv",index=False)
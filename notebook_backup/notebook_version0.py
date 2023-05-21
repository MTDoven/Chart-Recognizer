### Different path
YOLO_master = "C:\\Users\\t1526\\Desktop\\kaggle\\input\\ultralytics-main\\"
CRNN_master = "C:\\Users\\t1526\\Desktop\\kaggle\\input\\crnn-pytorch-master\\"
PATH_images = "C:\\Users\\t1526\\Desktop\\kaggle\\input\\benetech-making-graphs-accessible\\train\\images\\"
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
            'label_angle00','label_angle45','label_angle90','label_angle-45',
#                 5               6                 7               8
            'axis_x','axis_y','tick_point_x','tick_point_y','useless']
#               9       10          11             12           13

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





def IoU(box1, box2): # Intersection over Union
    x1,y1,x2,y2 = box1
    a1,b1,a2,b2 = box2
    ax=max(x1,a1); ay=max(y1,b1)
    bx=min(x2,a2); by=min(y2,b2)
    area_N = (x2-x1)*(y2-y1)
    area_M = (a2-a1)*(b2-b1)
    w=bx-ax; h=by-ay
    if w<=0 or h<=0: return 0
    area_X = w*h
    return area_X/(area_N+area_M-area_X)
def IoS(box1, box2): # Intersection over Smaller
    x1,y1,x2,y2 = box1
    a1,b1,a2,b2 = box2
    ax=max(x1,a1); ay=max(y1,b1)
    bx=min(x2,a2); by=min(y2,b2)
    area_N = (x2-x1)*(y2-y1)
    w=bx-ax; h=by-ay
    if w<=0 or h<=0: return 0
    area_X = w*h
    return area_X/area_N
def get_transform_function_y(label_ticks_y):
    label_ticks_y.sort(key=lambda x:x[1][1])
    def choose_great_label(label):
        crop_box = (label[1][0]-3,label[1][1],label[1][2]+3,label[1][3])
        cropped_img = img.crop(crop_box)
        words = IWR(cropped_img)
        words = words.replace(",", '');
        words = words.replace(" ", '');
        try: # avoid float changing problem
            if words[-1]=='%': words = float(words[:-1])
            elif words[0]=='$': words = float(words[1:])
            else: words = float(words)
        except: return "False"
        return words
    # pos1
    number = "False"
    times_count = 0
    while times_count<len(label_ticks_y):
        number = choose_great_label(label_ticks_y[times_count])
        if number!="False": break
        times_count+=1
    else: # give up
        print("dead circle in transform_y; float failure",information)
        raise CannotFixError
    picture_pos = (label_ticks_y[times_count][1][1]+
                   label_ticks_y[times_count][1][3])/2
    point1 = {"x":picture_pos, "y":number}
    # pos2
    number = "False"
    if len(label_ticks_y)<=4: times_count = 1
    else: times_count = 2
    while times_count<len(label_ticks_y):
        number = choose_great_label(label_ticks_y[len(label_ticks_y)-times_count])
        if number!="False": break
        times_count+=1
    else: # give up
        print("dead circle in transform_y; float failure",information)
        raise CannotFixError
    picture_pos = (label_ticks_y[len(label_ticks_y)-times_count][1][1]+
                   label_ticks_y[len(label_ticks_y)-times_count][1][3])/2
    point2 = {"x":picture_pos, "y":number}
    # check
    if abs(point1['x']-point2['x'])+abs(point1['y']-point2['y'])<0.001:
        print("not enough information to ensure axis",information)
        raise CannotFixError
    # get function
    k = (point2['y']-point1['y'])/(point2['x']-point1['x'])
    b = point1['y']-point1['x']*k
    return lambda x:k*x+b
def get_transform_function_x(label_ticks_y):
    label_ticks_y.sort(key=lambda x:x[1][2])
    def choose_great_label(label):
        crop_box = (label[1][0]-3,label[1][1]-2,label[1][2]+3,label[1][3]+2)
        cropped_img = img.crop(crop_box)
        words = IWR(cropped_img)
        words = words.replace(",", '');
        words = words.replace(" ", '');
        try: # avoid float changing problem
            if words[-1]=='%': words = float(words[:-1])
            elif words[0]=='$': words = float(words[1:])
            else: words = float(words)
        except: return "False"
        return words
    # pos1
    number = "False"
    times_count = 0
    while times_count<len(label_ticks_y):
        number = choose_great_label(label_ticks_y[times_count])
        if number!="False": break
        times_count+=1
    else: # give up
        print("dead circle in transform_x; float failure",information)
        raise CannotFixError
    picture_pos = (label_ticks_y[times_count][1][0]+
                   label_ticks_y[times_count][1][2])/2
    point1 = {"x":picture_pos, "y":number}
    # pos2
    number = "False"
    if len(label_ticks_y)<=4: times_count = 1
    else: times_count = 2
    while times_count<len(label_ticks_y):
        number = choose_great_label(label_ticks_y[len(label_ticks_y)-times_count])
        if number!="False": break
        times_count+=1
    else: # give up
        print("dead circle in transform_x; float failure",information)
        raise CannotFixError
    picture_pos = (label_ticks_y[len(label_ticks_y)-times_count][1][0]+
                   label_ticks_y[len(label_ticks_y)-times_count][1][2])/2
    point2 = {"x":picture_pos, "y":number}
    # check
    if abs(point1['x']-point2['x'])+abs(point1['y']-point2['y'])<0.001:
        print("not enough information to ensure axis",information)
        raise CannotFixError
    # get function
    k = (point2['y']-point1['y'])/(point2['x']-point1['x'])
    b = point1['y']-point1['x']*k
    return lambda x:k*x+b





class XAxisRepairError(AssertionError): pass
class CannotFixError(AssertionError): pass
class ChartTypeUnknownError(AssertionError): pass
class AxisUnknownError(AssertionError): pass





### start and start log
import time
log_exception = 0
log_chart_type = {'vertical_bar':0, 'horizontal_bar':0, 'dot':0, 'line':0, 'scatter':0}
time_start = time.time()

### main

for i, (img, information) in enumerate(dataloader):

    try:

        label_ticks = []
        label_ticks_x = []
        label_ticks_y = []
        tick_point_x = []
        tick_point_y = []
        chart_type_fixed = False
        x_axis_fixed = False
        y_axis_fixed = False
        flag = 'normal'
        chart_type = None
        result_df_writed = False

        ### object detect (ODT)
        resultODT = ODT(img)
        boxes = list(resultODT.cpu().numpy())
        # extract objects
        for obj in boxes:
            obj_class = int(obj[-1] + 0.5)
            if 5 <= obj_class <= 8:  # tick_label
                label_type = OBJ_TYPE[obj_class]
                label_box = list(map(lambda x: int(x + 0.5), obj[:4]))
                label_ticks.append((label_type, label_box))
            elif obj_class == 11:  # tick_point_x
                tick_point_x.append((obj[0] + obj[2]) / 2)
            elif obj_class == 12:  # tick_point_y
                tick_point_y.append((obj[1] + obj[3]) / 2)
            elif chart_type_fixed == False and 0 <= obj_class <= 4:  # chart_type
                chart_type = OBJ_TYPE[obj_class];
                chart_box = list(map(lambda x: int(x + 0.5), obj[0:4]))
                chart_type_fixed = True
            elif x_axis_fixed == False and obj_class == 9:  # x_axis
                x_axis_box = list(map(lambda x: int(x + 0.5), obj[0:4]))
                x_axis_fixed = True
            elif y_axis_fixed == False and obj_class == 10:  # y_axis
                y_axis_box = list(map(lambda x: int(x + 0.5), obj[0:4]))
                y_axis_fixed = True
            elif obj_class in [13, 9, 10, 0, 1, 2, 3, 4]:
                pass
            else:  # Unexpected label
                raise ValueError(f"Unexpected label '{obj_class}' exists")
        # tick_point_x; tick_point_y; x_axis_box; y_axis_box;
        # chart_type; chart_box; label_ticks;
        if chart_type is None:  # give up
            print("cannot ensure chart_type", information)
            raise ChartTypeUnknownError

        ### checking (check number of x-axis and y-axis)
        # if x&y_axis_fixed is usable or not
        for label in label_ticks:
            if x_axis_fixed == True and IoS(label[1], x_axis_box) > 0.667:
                label_ticks_x.append(label)
            elif y_axis_fixed == True and IoS(label[1], y_axis_box) > 0.667:
                label_ticks_y.append(label)
        if len(label_ticks_x) <= 1: x_axis_fixed == False
        if len(label_ticks_y) <= 1: y_axis_fixed == False
        if x_axis_fixed == True and y_axis_fixed == True:
            pass
        elif x_axis_fixed == False and y_axis_fixed == True:
            label_ticks_x.clear()
            for label in label_ticks:
                if label in label_ticks_y:
                    pass
                else:
                    label_ticks_x.append(label)
            x_axis_fixed = True
        elif x_axis_fixed == True and y_axis_fixed == False:
            label_ticks_y.clear()
            for label in label_ticks:
                if label in label_ticks_x:
                    pass
                else:
                    label_ticks_y.append(label)
            y_axis_fixed = True
        else:  # give up
            print("x_axis and y_axis are both not found", information)
            raise AxisUnknownError
        # add x-axis label in label_angle00
        if (chart_type in ['vertical_bar', 'line', 'dot']) and label_ticks_x[1][0] == 'label_angle00':
            label_ticks_x.sort(key=lambda x: x[1][2])
            tick_point_x.sort()
            tick_point_diatences = [tick_point_x[i + 1] - tick_point_x[i] for i in range(len(tick_point_x) - 1)]
            reliable_tick_point = max(tick_point_diatences) - min(tick_point_diatences) < 3
            centers = list(map(lambda x: (x[1][0] + x[1][2]) / 2, label_ticks_x))
            distance_min = min([centers[i + 1] - centers[i] for i in range(len(centers) - 1)])
            if reliable_tick_point and x_axis_fixed:
                ptr = min(tick_point_x[0], x_axis_box[0])
            elif reliable_tick_point:
                ptr = tick_point_x[0]
            elif x_axis_fixed:
                ptr = x_axis_box[0]
            elif chart_type_fixed:
                ptr = chart_box[0]
            else:
                ptr = centers[0]
            if (centers[0] - ptr) < distance_min:
                ptr = centers[0]
            else:  # (centers[0]-ptr)>=distance_min:
                blocks = (centers[0] - ptr) // distance_min
                ptr = centers[0] - blocks * (distance_min + 0.5)
            max_height = -1000000000
            min_height = 1000000000
            for _, label in label_ticks_x:
                if label[1] < min_height: min_height = label[1]
                if label[1] > max_height: max_height = label[3]
            times_count = 0
            while times_count < 100:  # break in circle
                times_count += 1
                for _, box in label_ticks_x:
                    if box[0] <= ptr <= box[2]:
                        ptr = (box[0] + box[2]) / 2 + distance_min + 0.5;
                        break
                else:  # no label box
                    box = [int(ptr - distance_min / 2 + 0.5), x_axis_box[1],  ###
                           int(ptr + distance_min / 2 + 0.5), x_axis_box[3]]  ###
                    label_ticks_x.append(('label_angle00', box))
                    ptr += (distance_min + 0.5)
                if reliable_tick_point and x_axis_fixed:
                    if ptr > (tick_point_x[-1] + distance_min / 2) and ptr > x_axis_box[2]: break
                elif reliable_tick_point:
                    if ptr > (tick_point_x[-1] + distance_min / 2): break
                elif x_axis_fixed:
                    if ptr > x_axis_box[2]: break
                elif chart_type_fixed:
                    if ptr > chart_box[2]: break
                else:  # give up
                    print("not confident enought to fix x-axis", information)
                    raise XAxisRepairError
            else:  # give up
                print("x-axis fix wrong in a dead circle", information)
                raise XAxisRepairError

        ### vertical_bar
        if chart_type == 'vertical_bar':
            log_chart_type['vertical_bar'] += 1
            transform_function = get_transform_function_y(label_ticks_y)
            label_ticks_x.sort(key=lambda x: x[1][2])
            words_height_list = []
            for label in label_ticks_x:
                crop_box = (label[1][0] - 3, label[1][1] - 3, label[1][2] + 3, label[1][3] + 3)
                cropped_img = img.crop(crop_box)
                cropped_img = cropped_img.resize((cropped_img.size[0] * 8, cropped_img.size[1] * 8))
                if label[0] == 'label_angle00':
                    pass  # need not to process
                elif label[0] == 'label_angle90':
                    cropped_img = cropped_img.rotate(-90)
                elif label[0] == 'label_angle45':
                    cropped_img = cropped_img.rotate(-45, expand=True)
                    crop_box = (60, int(cropped_img.size[1] / 2 - 70 + 0.5),
                                cropped_img.size[0] - 60, int(cropped_img.size[1] / 2 + 70 + 0.5))
                    cropped_img = cropped_img.crop(crop_box)
                elif label[0] == 'label_angle-45':
                    cropped_img = cropped_img.rotate(45, expand=True)
                    crop_box = (60, int(cropped_img.size[1] / 2 - 70 + 0.5),
                                cropped_img.size[0] - 60, int(cropped_img.size[1] / 2 + 70 + 0.5))
                    cropped_img = cropped_img.crop(crop_box)
                words = IWR(cropped_img)
                if label[0] == 'label_angle00' or label[0] == 'label_angle90':
                    center = (label[1][0] + label[1][2]) / 2
                elif label[0] == 'label_angle45':
                    center = label[1][2] - 5
                elif label[0] == 'label_angle-45':
                    center = label[1][0] + 5
                crop_box = (center - 6, chart_box[1] - 10, center + 6, chart_box[3])
                cropped_img = img.crop(crop_box)
                position = BHR(cropped_img)
                if len(position) == 0:
                    true_height = chart_box[3] - 0.75
                else:  # recognised
                    position = list(position[0].cpu().numpy()[:4])
                    true_height = (position[1] + position[3]) / 2 + chart_box[1] - 10
                true_height = transform_function(true_height)
                words_height_list.append([words, true_height])
            # x-series
            value1 = information[:-4] + "_x"
            value2 = ""
            for i in words_height_list:
                value2 += (i[0] + ";")
            value2 = value2[:-1]
            value3 = chart_type
            result_df.loc[len(result_df.index)] = [value1, value2, value3]
            # y-series
            value1 = information[:-4] + "_y"
            value2 = ""
            for i in words_height_list:
                value2 += (str(i[1]) + ";")
            value2 = value2[:-1]
            value3 = chart_type
            result_df.loc[len(result_df.index)] = [value1, value2, value3]
            result_df_writed = True

        ### horizontal_bar
        elif chart_type == 'horizontal_bar':
            log_chart_type['horizontal_bar'] += 1
            transform_function = get_transform_function_x(label_ticks_y)
            label_ticks_x.sort(key=lambda x: x[1][1])
            words_height_list = []
            for label in label_ticks_x:
                crop_box = (label[1][0] - 2, label[1][1] - 2, label[1][2] + 2, label[1][3] + 2)
                cropped_img = img.crop(crop_box)
                cropped_img = cropped_img.resize((cropped_img.size[0] * 8, cropped_img.size[1] * 8))
                words = IWR(cropped_img)
                center = (label[1][1] + label[1][3]) / 2
                crop_box = (chart_box[0], center - 6, chart_box[2], center + 6)
                cropped_img = img.crop(crop_box).transpose(Image.ROTATE_90)
                position = BHR(cropped_img)
                if len(position) == 0:
                    true_height = chart_box[3] - 0.75
                else:  # recognised
                    position = list(position[0].cpu().numpy()[:4])
                    true_height = cropped_img.size[1] - (position[1] + position[3]) / 2 + chart_box[0]
                true_height = transform_function(true_height)
                words_height_list.append([words, true_height])
            # x-series
            value1 = information[:-4] + "_x"
            value2 = ""
            for i in words_height_list:
                value2 += (str(i[1]) + ";")
            value2 = value2[:-1]
            value3 = chart_type
            result_df.loc[len(result_df.index)] = [value1, value2, value3]
            # y-series
            value1 = information[:-4] + "_y"
            value2 = ""
            for i in words_height_list:
                value2 += (i[0] + ";")
            value2 = value2[:-1]
            value3 = chart_type
            result_df.loc[len(result_df.index)] = [value1, value2, value3]
            result_df_writed = True

        ### line
        elif chart_type == 'line':
            log_chart_type['line'] += 1
            transform_function = get_transform_function_y(label_ticks_y)
            label_ticks_x.sort(key=lambda x: x[1][2])
            words_height_list = []
            for label in label_ticks_x:
                crop_box = (label[1][0] - 3, label[1][1] - 3, label[1][2] + 3, label[1][3] + 3)
                cropped_img = img.crop(crop_box)
                cropped_img = cropped_img.resize((cropped_img.size[0] * 8, cropped_img.size[1] * 8))
                if label[0] == 'label_angle00':
                    pass  # need not to process
                elif label[0] == 'label_angle90':
                    cropped_img = cropped_img.rotate(-90)
                elif label[0] == 'label_angle45':
                    cropped_img = cropped_img.rotate(-45, expand=True)
                    crop_box = (60, int(cropped_img.size[1] / 2 - 70 + 0.5),
                                cropped_img.size[0] - 60, int(cropped_img.size[1] / 2 + 70 + 0.5))
                    cropped_img = cropped_img.crop(crop_box)
                elif label[0] == 'label_angle-45':
                    cropped_img = cropped_img.rotate(45, expand=True)
                    crop_box = (60, int(cropped_img.size[1] / 2 - 70 + 0.5),
                                cropped_img.size[0] - 60, int(cropped_img.size[1] / 2 + 70 + 0.5))
                    cropped_img = cropped_img.crop(crop_box)
                words = IWR(cropped_img)
                if label[0] == 'label_angle00' or label[0] == 'label_angle90':
                    center = (label[1][0] + label[1][2]) / 2
                elif label[0] == 'label_angle45':
                    center = label[1][2] - 5
                elif label[0] == 'label_angle-45':
                    center = label[1][0] + 5
                crop_box = (center - 16, chart_box[1] - 10, center + 16, chart_box[3])
                cropped_img = img.crop(crop_box)
                # cropped_img.show()
                position = LHR(cropped_img)
                # print(position)
                if len(position) == 0:
                    true_height = chart_box[3] - 0.75
                else:  # recognised
                    position = list(position[0].cpu().numpy()[:4])
                    true_height = (position[1] + position[3]) / 2 + chart_box[1] - 10
                true_height = transform_function(true_height)
                words_height_list.append([words, true_height])
            # x-series
            value1 = information[:-4] + "_x"
            value2 = ""
            for i in words_height_list:
                value2 += (i[0] + ";")
            value2 = value2[:-1]
            value3 = chart_type
            result_df.loc[len(result_df.index)] = [value1, value2, value3]
            # y-series
            value1 = information[:-4] + "_y"
            value2 = ""
            for i in words_height_list:
                value2 += (str(i[1]) + ";")
            value2 = value2[:-1]
            value3 = chart_type
            result_df.loc[len(result_df.index)] = [value1, value2, value3]
            result_df_writed = True

        ### dot
        elif chart_type == 'dot':
            log_chart_type['dot'] += 1
            label_ticks_x.sort(key=lambda x: x[1][2])
            words_height_list = []
            for label in label_ticks_x:
                crop_box = (label[1][0] - 3, label[1][1] - 3, label[1][2] + 3, label[1][3] + 3)
                cropped_img = img.crop(crop_box)
                cropped_img = cropped_img.resize((cropped_img.size[0] * 8, cropped_img.size[1] * 8))
                if label[0] == 'label_angle00':
                    pass  # need not to process
                elif label[0] == 'label_angle90':
                    cropped_img = cropped_img.rotate(-90)
                elif label[0] == 'label_angle45':
                    cropped_img = cropped_img.rotate(-45, expand=True)
                    crop_box = (60, int(cropped_img.size[1] / 2 - 70 + 0.5),
                                cropped_img.size[0] - 60, int(cropped_img.size[1] / 2 + 70 + 0.5))
                    cropped_img = cropped_img.crop(crop_box)
                elif label[0] == 'label_angle-45':
                    cropped_img = cropped_img.rotate(45, expand=True)
                    crop_box = (60, int(cropped_img.size[1] / 2 - 70 + 0.5),
                                cropped_img.size[0] - 60, int(cropped_img.size[1] / 2 + 70 + 0.5))
                    cropped_img = cropped_img.crop(crop_box)
                words = IWR(cropped_img)
                if label[0] == 'label_angle00' or label[0] == 'label_angle90':
                    center = (label[1][0] + label[1][2]) / 2
                elif label[0] == 'label_angle45':
                    center = label[1][2] - 5
                elif label[0] == 'label_angle-45':
                    center = label[1][0] + 5
                crop_box = (center - 10, chart_box[1] - 5, center + 10, chart_box[3])
                cropped_img = img.crop(crop_box)
                number = DNR(cropped_img)
                words_height_list.append([words, number])
            # x-series
            value1 = information[:-4] + "_x"
            value2 = ""
            for i in words_height_list:
                value2 += (i[0] + ";")
            value2 = value2[:-1]
            value3 = chart_type
            result_df.loc[len(result_df.index)] = [value1, value2, value3]
            # y-series
            value1 = information[:-4] + "_y"
            value2 = ""
            for i in words_height_list:
                value2 += (str(i[1]) + ";")
            value2 = value2[:-1]
            value3 = chart_type
            result_df.loc[len(result_df.index)] = [value1, value2, value3]
            result_df_writed = True

        ### scatter
        elif chart_type == 'scatter':
            log_chart_type['scatter'] += 1
            transform_function_y = get_transform_function_y(label_ticks_y)
            transform_function_x = get_transform_function_x(label_ticks_x)
            scatters = STR(img)
            processed_scatters = []
            for scatter in scatters.cpu().numpy():
                x = (scatter[0] + scatter[2]) / 2
                y = (scatter[1] + scatter[3]) / 2
                if not (chart_box[0] - 7 < x < chart_box[2] + 7): continue
                if not (chart_box[1] - 7 < y < chart_box[3] + 7): continue
                x = transform_function_x(x)
                y = transform_function_y(y)
                processed_scatters.append([x, y])
            processed_scatters.sort(key=lambda x: x[0])
            # x-series
            value1 = information[:-4] + "_x"
            value2 = ""
            for i in processed_scatters:
                value2 += (str(i[0]) + ";")
            value2 = value2[:-1]
            value3 = chart_type
            result_df.loc[len(result_df.index)] = [value1, value2, value3]
            # y-series
            value1 = information[:-4] + "_y"
            value2 = ""
            for i in processed_scatters:
                value2 += (str(i[1]) + ";")
            value2 = value2[:-1]
            value3 = chart_type
            result_df.loc[len(result_df.index)] = [value1, value2, value3]
            result_df_writed = True

        ### something else
        else:  # chart_type unkown
            print("cannot ensure chart_type", information)
            raise ChartTypeUnknownError

    ### exception process
    # XAxisRepairError
    except (ZeroDivisionError, OverflowError, XAxisRepairError):
        print("try to skip XAxisRepair", information)
        try:  # to avoid any other problem
            label_ticks_x.clear()
            label_ticks_y.clear()
            # if x&y_axis_fixed is usable or not
            for label in label_ticks:
                if x_axis_fixed == True and IoS(label[1], x_axis_box) > 0.667:
                    label_ticks_x.append(label)
                elif y_axis_fixed == True and IoS(label[1], y_axis_box) > 0.667:
                    label_ticks_y.append(label)
            if len(label_ticks_x) <= 1: x_axis_fixed == False
            if len(label_ticks_y) <= 1: y_axis_fixed == False
            if x_axis_fixed == True and y_axis_fixed == True:
                pass
            elif x_axis_fixed == False and y_axis_fixed == True:
                label_ticks_x.clear()
                for label in label_ticks:
                    if label in label_ticks_y:
                        pass
                    else:
                        label_ticks_x.append(label)
                x_axis_fixed = True
            elif x_axis_fixed == True and y_axis_fixed == False:
                label_ticks_y.clear()
                for label in label_ticks:
                    if label in label_ticks_x:
                        pass
                    else:
                        label_ticks_y.append(label)
                y_axis_fixed = True
            else:  # give up
                print("x_axis and y_axis are both not found", information)
                raise AxisUnknownError
            ### vertical_bar
            if chart_type == 'vertical_bar':
                log_chart_type['vertical_bar'] += 1
                transform_function = get_transform_function_y(label_ticks_y)
                label_ticks_x.sort(key=lambda x: x[1][2])
                words_height_list = []
                for label in label_ticks_x:
                    crop_box = (label[1][0] - 3, label[1][1] - 3, label[1][2] + 3, label[1][3] + 3)
                    cropped_img = img.crop(crop_box)
                    cropped_img = cropped_img.resize((cropped_img.size[0] * 8, cropped_img.size[1] * 8))
                    if label[0] == 'label_angle00':
                        pass  # need not to process
                    elif label[0] == 'label_angle90':
                        cropped_img = cropped_img.rotate(-90)
                    elif label[0] == 'label_angle45':
                        cropped_img = cropped_img.rotate(-45, expand=True)
                        crop_box = (60, int(cropped_img.size[1] / 2 - 70 + 0.5),
                                    cropped_img.size[0] - 60, int(cropped_img.size[1] / 2 + 70 + 0.5))
                        cropped_img = cropped_img.crop(crop_box)
                    elif label[0] == 'label_angle-45':
                        cropped_img = cropped_img.rotate(45, expand=True)
                        crop_box = (60, int(cropped_img.size[1] / 2 - 70 + 0.5),
                                    cropped_img.size[0] - 60, int(cropped_img.size[1] / 2 + 70 + 0.5))
                        cropped_img = cropped_img.crop(crop_box)
                    words = IWR(cropped_img)
                    if label[0] == 'label_angle00' or label[0] == 'label_angle90':
                        center = (label[1][0] + label[1][2]) / 2
                    elif label[0] == 'label_angle45':
                        center = label[1][2] - 5
                    elif label[0] == 'label_angle-45':
                        center = label[1][0] + 5
                    crop_box = (center - 6, chart_box[1] - 10, center + 6, chart_box[3])
                    cropped_img = img.crop(crop_box)
                    position = BHR(cropped_img)
                    if len(position) == 0:
                        true_height = chart_box[3] - 0.75
                    else:  # recognised
                        position = list(position[0].cpu().numpy()[:4])
                        true_height = (position[1] + position[3]) / 2 + chart_box[1] - 10
                    true_height = transform_function(true_height)
                    words_height_list.append([words, true_height])
                # x-series
                value1 = information[:-4] + "_x"
                value2 = ""
                for i in words_height_list:
                    value2 += (i[0] + ";")
                value2 = value2[:-1]
                value3 = chart_type
                result_df.loc[len(result_df.index)] = [value1, value2, value3]
                # y-series
                value1 = information[:-4] + "_y"
                value2 = ""
                for i in words_height_list:
                    value2 += (str(i[1]) + ";")
                value2 = value2[:-1]
                value3 = chart_type
                result_df.loc[len(result_df.index)] = [value1, value2, value3]
                result_df_writed = True
            ### line
            elif chart_type == 'line':
                log_chart_type['line'] += 1
                transform_function = get_transform_function_y(label_ticks_y)
                label_ticks_x.sort(key=lambda x: x[1][2])
                words_height_list = []
                for label in label_ticks_x:
                    crop_box = (label[1][0] - 3, label[1][1] - 3, label[1][2] + 3, label[1][3] + 3)
                    cropped_img = img.crop(crop_box)
                    cropped_img = cropped_img.resize((cropped_img.size[0] * 8, cropped_img.size[1] * 8))
                    if label[0] == 'label_angle00':
                        pass  # need not to process
                    elif label[0] == 'label_angle90':
                        cropped_img = cropped_img.rotate(-90)
                    elif label[0] == 'label_angle45':
                        cropped_img = cropped_img.rotate(-45, expand=True)
                        crop_box = (60, int(cropped_img.size[1] / 2 - 70 + 0.5),
                                    cropped_img.size[0] - 60, int(cropped_img.size[1] / 2 + 70 + 0.5))
                        cropped_img = cropped_img.crop(crop_box)
                    elif label[0] == 'label_angle-45':
                        cropped_img = cropped_img.rotate(45, expand=True)
                        crop_box = (60, int(cropped_img.size[1] / 2 - 70 + 0.5),
                                    cropped_img.size[0] - 60, int(cropped_img.size[1] / 2 + 70 + 0.5))
                        cropped_img = cropped_img.crop(crop_box)
                    words = IWR(cropped_img)
                    if label[0] == 'label_angle00' or label[0] == 'label_angle90':
                        center = (label[1][0] + label[1][2]) / 2
                    elif label[0] == 'label_angle45':
                        center = label[1][2] - 5
                    elif label[0] == 'label_angle-45':
                        center = label[1][0] + 5
                    crop_box = (center - 16, chart_box[1] - 10, center + 16, chart_box[3])
                    cropped_img = img.crop(crop_box)
                    # cropped_img.show()
                    position = LHR(cropped_img)
                    # print(position)
                    if len(position) == 0:
                        true_height = chart_box[3] - 0.75
                    else:  # recognised
                        position = list(position[0].cpu().numpy()[:4])
                        true_height = (position[1] + position[3]) / 2 + chart_box[1] - 10
                    true_height = transform_function(true_height)
                    words_height_list.append([words, true_height])
                # x-series
                value1 = information[:-4] + "_x"
                value2 = ""
                for i in words_height_list:
                    value2 += (i[0] + ";")
                value2 = value2[:-1]
                value3 = chart_type
                result_df.loc[len(result_df.index)] = [value1, value2, value3]
                # y-series
                value1 = information[:-4] + "_y"
                value2 = ""
                for i in words_height_list:
                    value2 += (str(i[1]) + ";")
                value2 = value2[:-1]
                value3 = chart_type
                result_df.loc[len(result_df.index)] = [value1, value2, value3]
                result_df_writed = True
            ### dot
            elif chart_type == 'dot':
                log_chart_type['dot'] += 1
                label_ticks_x.sort(key=lambda x: x[1][2])
                words_height_list = []
                for label in label_ticks_x:
                    crop_box = (label[1][0] - 3, label[1][1] - 3, label[1][2] + 3, label[1][3] + 3)
                    cropped_img = img.crop(crop_box)
                    cropped_img = cropped_img.resize((cropped_img.size[0] * 8, cropped_img.size[1] * 8))
                    if label[0] == 'label_angle00':
                        pass  # need not to process
                    elif label[0] == 'label_angle90':
                        cropped_img = cropped_img.rotate(-90)
                    elif label[0] == 'label_angle45':
                        cropped_img = cropped_img.rotate(-45, expand=True)
                        crop_box = (60, int(cropped_img.size[1] / 2 - 70 + 0.5),
                                    cropped_img.size[0] - 60, int(cropped_img.size[1] / 2 + 70 + 0.5))
                        cropped_img = cropped_img.crop(crop_box)
                    elif label[0] == 'label_angle-45':
                        cropped_img = cropped_img.rotate(45, expand=True)
                        crop_box = (60, int(cropped_img.size[1] / 2 - 70 + 0.5),
                                    cropped_img.size[0] - 60, int(cropped_img.size[1] / 2 + 70 + 0.5))
                        cropped_img = cropped_img.crop(crop_box)
                    words = IWR(cropped_img)
                    if label[0] == 'label_angle00' or label[0] == 'label_angle90':
                        center = (label[1][0] + label[1][2]) / 2
                    elif label[0] == 'label_angle45':
                        center = label[1][2] - 5
                    elif label[0] == 'label_angle-45':
                        center = label[1][0] + 5
                    crop_box = (center - 10, chart_box[1] - 5, center + 10, chart_box[3])
                    cropped_img = img.crop(crop_box)
                    number = DNR(cropped_img)
                    words_height_list.append([words, number])
                # x-series
                value1 = information[:-4] + "_x"
                value2 = ""
                for i in words_height_list:
                    value2 += (i[0] + ";")
                value2 = value2[:-1]
                value3 = chart_type
                result_df.loc[len(result_df.index)] = [value1, value2, value3]
                # y-series
                value1 = information[:-4] + "_y"
                value2 = ""
                for i in words_height_list:
                    value2 += (str(i[1]) + ";")
                value2 = value2[:-1]
                value3 = chart_type
                result_df.loc[len(result_df.index)] = [value1, value2, value3]
                result_df_writed = True
        except:  # anything wrong
            print('fail to fix XAxisRepairError', information)
        log_exception += 1
    except ChartTypeUnknownError:
        print("try to fix ChartTypeUnknownError", information)
        chart_type = 'scatter'
        try:  # to avoid other problem
            transform_function_y = get_transform_function_y(label_ticks_y)
            transform_function_x = get_transform_function_x(label_ticks_x)
            scatters = STR(img)
            processed_scatters = []
            for scatter in scatters.cpu().numpy():
                x = (scatter[0] + scatter[2]) / 2
                y = (scatter[1] + scatter[3]) / 2
                if not (chart_box[0] - 7 < x < chart_box[2] + 7): continue
                if not (chart_box[1] - 7 < y < chart_box[3] + 7): continue
                x = transform_function_x(x)
                y = transform_function_y(y)
                processed_scatters.append([x, y])
            processed_scatters.sort(key=lambda x: x[0])
            # x-series
            value1 = information[:-4] + "_x"
            value2 = ""
            for i in processed_scatters:
                value2 += (str(i[0]) + ";")
            value2 = value2[:-1]
            value3 = chart_type
            result_df.loc[len(result_df.index)] = [value1, value2, value3]
            # y-series
            value1 = information[:-4] + "_y"
            value2 = ""
            for i in processed_scatters:
                value2 += (str(i[1]) + ";")
            value2 = value2[:-1]
            value3 = chart_type
            result_df.loc[len(result_df.index)] = [value1, value2, value3]
            result_df_writed = True
        except:  # anything wrong
            print('fail to fix ChartTypeUnknownError', information)
        log_exception += 1
    except AxisUnknownError:
        print("not to fix AxisUnknownError", information)
        log_exception += 1
    except CannotFixError:
        print("this is a CannotFixError", information)
        log_exception += 1
    except Exception as error:
        print('Unexpected error: ', error.__class__.__name__, error, information)
        log_exception += 1
    finally:
        if result_df_writed == True:
            pass  # have writed
        else:  # have not writed
            value3 = chart_type
            value1 = information[:-4] + "_x"
            result_df.loc[len(result_df.index)] = [value1, "unknown", value3]
            value1 = information[:-4] + "_y"
            result_df.loc[len(result_df.index)] = [value1, "unknown", value3]
            result_df_writed = True






### save
result_df.to_csv("./submission.csv",index=False)





### end and print log
print('!!!___finished___!!!')
print("log_exception:",log_exception)
print("data_length:",len(dataloader))
print(log_chart_type)
print("time:", (time.time()-time_start)/3600)






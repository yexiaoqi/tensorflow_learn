import numpy as np
import sys
import time


# https://blog.csdn.net/qq_21997625/article/details/85406153
def non_max_suppress(predicts_dict, threshold=0.2):
    for object_name, bbox in predicts_dict.items():
        #bbox是tuple,tuple不能直接像下面第二行一样分割
        bbox_array = np.array(bbox, dtype=np.float)
        x1, y1, x2, y2, scores = bbox_array[:, 0], bbox_array[:, 1], bbox_array[:, 2], bbox_array[:, 3], bbox_array[:,
                                                                                                         4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[:: - 1]#注意argsort要有()
        keep = []

        # size不应该带括号，带括号的是np.size(order)
        # while np.size(order)>0:
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            print("iou=", iou)
            indexs = np.where(iou <= threshold)[0] + 1  # 获取保留下来的索引(因为没有计算与自身的IOU，所以索引相差１，需要加上)
            order = order[indexs]

        bbox = bbox_array[keep]
        predicts_dict[object_name] = bbox.tolist()

    return predicts_dict


if __name__ == "__main__":
    box1 = (13, 22, 268, 367, 0.124648176)
    box2 = (18, 27, 294, 400, 0.35818103)
    box3 = (234, 123, 466, 678, 0.13638769)
    box_lists = [box1, box2, box3]
    predicts_dict = {'cat': box_lists}
    result = non_max_suppress(predicts_dict)
    print(result)

import numpy as np
import sys
import time
def non_max_suppress(predicts_dict, threshold = 0.2) :
	for object_name, bbox in predicts_dict.items() :
		bbox_array = np.array(bbox, dtype = np.float)
		x1, y1, x2, y2, scores = bbox_array[:, 0], bbox_array[:, 1], bbox_array[:, 2], bbox_array[:, 3], bbox_array[:4]
		areas = (x2 - x1 + 1)*(y2 - y1 + 1)
		order = scores.argsort()[:: - 1]
		keep = []
		while order.size()>0:
			i = order[0]
			keep.append(i)
			xx1 = np.maximum(x1[i], x1[order[1:]])
			yy1 = np.maximum(y1[i], y1[order[1:]])
			xx2 = np.minimum(x2[i], x2[order[1:]])
			yy2 = np.minimum(y2[i], y2[order[1:]])
			inter = np.maximum(0.0, xx2 - xx1 + 1)*np.maximum(0.0, yy2 - yy1 + 1)
			iou = inter / (areas[i] + areas[order[1:]] - inter)
			indexs = np.where(iou <= threshold)[0] + 1
			order = order[indexs]
		bbox = bbox_array[keep]
		predicts_dict[object_name] = bbox.tolist()
	return predicts_dict

print('this is nms')
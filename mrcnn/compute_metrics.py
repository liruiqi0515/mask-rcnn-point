import numpy as np

def compute_iou(box, boxes):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
#     print(boxes_area.shape)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    print(iou)
    return iou


def find_match(det_boxes, box):
    """
    Input:
    box: [4]
    det_boxes: [N,4]
    """
    iou = compute_iou(box, det_boxes)
    indice = np.argmax(iou)
    if iou[indice] > 0.5:
        return indice
    else:
        return None

def _compute_pre(pred, point, det_boxes, gt_bbox, threshold):
    """
    Input:
    box: [N,4]
    map: [N,x,y,num_keypoint]

    """
    num_joint = point.shape[1]
    correct_joint = np.zeros([num_joint])
    image_num = point.shape[0]
    pred_num = det_boxes.shape[0]
    # print(pred_num)
    if pred_num == 0:
        return correct_joint, image_num
    for i in range(image_num):
        box = gt_bbox[i]
        pred_i = find_match(det_boxes, box)
        if pred_i is not None:
            for joint in range(num_joint):
                pred_x, pred_y = np.unravel_index(pred[pred_i, :, :, joint].argmax(), pred[pred_i, :, :, joint].shape)
                pred_point = np.array([pred_y, pred_x])
                gtmap_point = point[i, joint] * pred.shape[1]
                if np.square(pred_point - gtmap_point).sum() <= threshold * threshold:
                    correct_joint[joint] += 1
        # print(correct_joint)
    return correct_joint, image_num
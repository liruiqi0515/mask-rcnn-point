import keras.backend as K
import cv2, time, os
import numpy as np
import model as modellib
from skimage import morphology


class MAPCallback:
    def __init__(self,
                 model,
                 val_dataset,
                 class_names,
                 threshold=5,
                 inference_num=50,
                 batch_size=1,
                 old_version=False):
        super(MAPCallback, self).__init__()
        self.model = model
        self.inference_num = inference_num
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.val_dataset = val_dataset
        self.threshold = threshold
        self.batch_size = batch_size
        self.old_version = old_version

    def _voc_ap(self, rec, prec):
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def calculate_result(self):
        true_res = {}
        pred_res = []
        inference_time = 0
        for i in range(self.inference_num):
            image, class_ids, bbox, point = modellib.load_image_gt_eval(self.val_dataset, i)
            start = time.time()
            results = self.model.detect([image])[0]
            end = time.time()
            inference_time = inference_time + (end - start)
            out_boxes = results['rois']
            out_scores = results['scores']
            out_masks = results['masks']
            pred_res_0 = []
            pred_res_1 = []
            if len(out_boxes) > 0:
                for out_box, out_score, out_mask in zip(
                        out_boxes, out_scores, out_masks):
                    det_point = np.unravel_index(out_mask[:, :, 0].argmax(), out_mask[:, :, 0].shape)
                    if self.old_version:
                        pred_res_0.append([i, 0, out_score, det_point[1] + 1, det_point[0] + 1])
                    else:
                        pred_res_0.append([i, 0, out_score * out_mask[:, :, 0].max(), det_point[1] + 1, det_point[0] + 1])
                    # print([i, 0, out_mask[:, :, 0].max(), det_point[1] + 1, det_point[0] + 1])
                    det_point = np.unravel_index(out_mask[:, :, 1].argmax(), out_mask[:, :, 1].shape)
                    if self.old_version:
                        pred_res_1.append([i, 1, out_score, det_point[1] + 1, det_point[0] + 1])
                    else:
                        pred_res_1.append([i, 1, out_score * out_mask[:, :, 1].max(), det_point[1] + 1, det_point[0] + 1])
                    # print([i, 1, out_score * out_mask[:, :, 1].max(), det_point[1] + 1, det_point[0] + 1])
            pred_res_0 = nms_point(pred_res_0, 10)
            pred_res_1 = nms_point(pred_res_1, 10)
            pred_res.extend(pred_res_0)
            pred_res.extend(pred_res_1)
            true_res[i] = point  # [num_guidewire, num_point, 2]
            # print(point)
        print('avg_infer_time:' + str(inference_time / self.inference_num))
        return true_res, pred_res

    def compute_aps(self, true_res, pred_res, threshold):
        APs = {}
        for cls in range(self.num_classes):
            pred_res_cls = [x for x in pred_res if x[1] == cls]
            if len(pred_res_cls) == 0:
                APs[cls] = 0
                continue
            true_res_cls = {}
            npos = 0
            for index in true_res:  # index is the image_id
                guidewires = true_res[index]  # [num_guidewire, num_point, 2]
                npos += len(guidewires)  # compute recall
                point_pos = np.array([x[cls] for x in guidewires])  # [num_guidewire, 2]
                true_res_cls[index] = {
                    'point_pos': point_pos,
                }
            ids = [x[0] for x in pred_res_cls]
            scores = np.array([x[2] for x in pred_res_cls])
            points = np.array([x[3:] for x in pred_res_cls])
            sorted_ind = np.argsort(-scores)
            points = points[sorted_ind, :]  # sorted
            ids = [ids[x] for x in sorted_ind]  # sorted

            nd = len(ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for j in range(nd):
                ture_point = true_res_cls[ids[j]]
                point1 = points[j, :]  # [2]
                dis_min = np.inf
                PGT = ture_point['point_pos']  # [num_guidewire, 2]
                if len(PGT) > 0:
                    dis_square = np.square(PGT[:, 0] - point1[0]) + np.square(PGT[:, 1] - point1[1])
                    dis_min = np.min(dis_square)
                if dis_min < threshold * threshold:
                    tp[j] = 1.
                else:
                    fp[j] = 1.

            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / np.maximum(float(npos), np.finfo(np.float64).eps)
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self._voc_ap(rec, prec)
            APs[cls] = ap
        return APs

    def on_epoch_end(self, logs=None):
        logs = logs or {}
        K.set_learning_phase(0)
        true_res, pred_res = self.calculate_result()
        for th in [3, 5, 7, 9]:
            APs = self.compute_aps(true_res, pred_res, th)
            for cls in range(self.num_classes):
                if cls in APs:
                    print(self.class_names[cls] + ' ap: ', APs[cls])
            mAP = np.mean([APs[cls] for cls in APs])
            print('mAP: ', mAP)
            logs['mAP'] = mAP


def nms_point(point_list, thresh):
    '''point_list: [i, point_id, score, x, y]'''
    keep = []
    while point_list:
        keep.append(point_list[0])
        now = point_list[0]
        del point_list[0]
        del_inds = []
        for i in range(len(point_list)):
            dis_square = np.square(point_list[i][3] - now[3]) + np.square(point_list[i][4] - now[4])
            if dis_square < thresh * thresh:
                del_inds.append(i)
        if del_inds:
            del_inds.reverse()
            for i in del_inds:
                del point_list[i]
    return keep


class MAPCallbackSame(MAPCallback):
    def __init__(self,
                 model,
                 val_dataset,
                 class_names,
                 threshold=5,
                 inference_num=50,
                 batch_size=1):
        super(MAPCallbackSame, self).__init__()
        self.model = model
        self.inference_num = inference_num
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.val_dataset = val_dataset
        self.threshold = threshold
        self.batch_size = batch_size

    def compute_point(self, pred, thresh, sigma):
        point = -1 * np.ones((2, 2), np.int32)
        idx = np.unravel_index(pred.argmax(), pred.shape)
        # print(pred.shape)
        if pred[idx[0], idx[1]] > thresh:
            point[0] = [idx[0], idx[1]]
            minus = makeGaussian(pred.shape[0], pred.shape[1], sigma, (idx[1], idx[0])) * pred[idx[0], idx[1]]
            pred = pred - minus
            idx_1 = np.unravel_index(pred.argmax(), pred.shape)
            if pred[idx_1[0], idx_1[1]] > thresh:
                point[1] = [idx_1[0], idx_1[1]]
        return point

    def calculate_result(self):
        true_res = {}
        pred_res = []
        inference_time = 0
        for i in range(self.inference_num):
            image, class_ids, bbox, point = modellib.load_image_gt_eval(self.val_dataset, i)
            start = time.time()
            results = self.model.detect([image])[0]
            end = time.time()
            inference_time = inference_time + (end - start)
            out_boxes = results['rois']
            out_scores = results['scores']
            out_masks = results['masks']
            if len(out_boxes) > 0:
                for out_box, out_score, out_mask in zip(
                        out_boxes, out_scores, out_masks):
                    det_point = self.compute_point(out_mask[:, :, 0], 0.1, 6)
                    pred_res.append([i, 0, out_score, det_point[0][1] + 1, det_point[0][0] + 1])
                    pred_res.append([i, 0, out_score, det_point[1][1] + 1, det_point[1][0] + 1])
                    # print([i, 0, out_score, det_point[0][1], det_point[0][0]])
                    # print([i, 0, out_score, det_point[1][1], det_point[1][0]])
            true_res[i] = point  # [num_guidewire, num_point, 2]
        print('avg_infer_time:' + str(inference_time / self.inference_num))
        return true_res, pred_res

    def compute_aps(self, true_res, pred_res, threshold):
        APs = {}
        for cls in range(self.num_classes):
            pred_res_cls = [x for x in pred_res if x[1] == cls]
            if len(pred_res_cls) == 0:
                APs[cls] = 0
                continue
            true_res_cls = {}
            npos = 0
            for index in true_res:  # index is the image_id
                guidewires = true_res[index]  # [num_guidewire, num_point, 2]
                guidewires = np.reshape(guidewires, [guidewires.shape[0] * guidewires.shape[1], 1, 2])
                npos += len(guidewires)  # compute recall
                point_pos = np.array([x[cls] for x in guidewires])  # [num_guidewire, 2]
                true_res_cls[index] = {
                    'point_pos': point_pos,
                }
            ids = [x[0] for x in pred_res_cls]
            scores = np.array([x[2] for x in pred_res_cls])
            points = np.array([x[3:] for x in pred_res_cls])
            sorted_ind = np.argsort(-scores)
            points = points[sorted_ind, :]  # sorted
            ids = [ids[x] for x in sorted_ind]  # sorted

            nd = len(ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for j in range(nd):
                ture_point = true_res_cls[ids[j]]
                point1 = points[j, :]  # [2]
                dis_min = np.inf
                PGT = ture_point['point_pos']  # [num_guidewire, 2]
                if len(PGT) > 0:
                    dis_square = np.square(PGT[:, 0] - point1[0]) + np.square(PGT[:, 1] - point1[1])
                    dis_min = np.min(dis_square)
                if dis_min < threshold * threshold:
                    tp[j] = 1.
                else:
                    fp[j] = 1.

            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / np.maximum(float(npos), np.finfo(np.float64).eps)
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self._voc_ap(rec, prec)
            APs[cls] = ap
        return APs


def makeGaussian(height, width, sigma=3, center=None):
    """ make一个高斯核，是生成heatmap的一个部分
    """
    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)[:, np.newaxis]
    if center is None:
        x0 = width // 2
        y0 = height // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / (sigma ** 2))


class MAPCallbackMask(MAPCallbackSame):
    def __init__(self,
                 model,
                 val_dataset,
                 class_names,
                 threshold=0.1,
                 inference_num=50,
                 batch_size=1):
        # super(MAPCallbackMask, self).__init__()
        self.model = model
        self.inference_num = inference_num
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.val_dataset = val_dataset
        self.threshold = threshold
        self.batch_size = batch_size

    def compute_point_from_mask(self, pred, thresh):
        pred = (pred > thresh).astype('uint8')
        skeleton = morphology.skeletonize(pred)
        fil = np.array([[1, 1, 1], [1, 8, 1], [1, 1, 1]])
        conv = cv2.filter2D(np.float32(skeleton), -1, fil)
        result = conv == 9
        x, y = np.where(result == True)
        endpoint = []
        num_point = min(len(x), 2)
        for i in range(num_point):
            endpoint.append(np.array([x[i], y[i]]))
        return endpoint

    def calculate_result(self):
        true_res = {}
        pred_res = []
        inference_time = 0
        for i in range(self.inference_num):
            image, class_ids, bbox, point = modellib.load_image_gt_eval(self.val_dataset, i)
            start = time.time()
            results = self.model.detect([image])[0]
            end = time.time()
            inference_time = inference_time + (end - start)
            out_boxes = results['rois']
            out_scores = results['scores']
            out_masks = results['masks']
            if len(out_boxes) > 0:
                for out_box, out_score, out_mask in zip(
                        out_boxes, out_scores, out_masks):
                    det_point = self.compute_point_from_mask(out_mask[:, :, 0], self.threshold)
                    for det_point_i in det_point:
                        pred_res.append([i, 0, out_score, det_point_i[1] + 1, det_point_i[0] + 1])
                    # print([i, 0, out_score, det_point[0][1], det_point[0][0]])
                    # print([i, 0, out_score, det_point[1][1], det_point[1][0]])
            true_res[i] = point  # [num_guidewire, num_point, 2]
            # print(point)
        print('avg_infer_time:' + str(inference_time / self.inference_num))
        return true_res, pred_res

    def compute_aps(self, true_res, pred_res, threshold):
        APs = {}
        for cls in range(self.num_classes):
            pred_res_cls = [x for x in pred_res if x[1] == cls]
            if len(pred_res_cls) == 0:
                APs[cls] = 0
                continue
            true_res_cls = {}
            npos = 0
            for index in true_res:  # index is the image_id
                guidewires = true_res[index]  # [num_guidewire, num_point, 2]
                guidewires = np.reshape(guidewires, [guidewires.shape[0] * guidewires.shape[1], 1, 2])
                npos += len(guidewires)  # compute recall
                point_pos = np.array([x[cls] for x in guidewires])  # [num_guidewire, 2]
                true_res_cls[index] = {
                    'point_pos': point_pos,
                }
            ids = [x[0] for x in pred_res_cls]
            scores = np.array([x[2] for x in pred_res_cls])
            points = np.array([x[3:] for x in pred_res_cls])
            sorted_ind = np.argsort(-scores)
            points = points[sorted_ind, :]  # sorted
            ids = [ids[x] for x in sorted_ind]  # sorted

            nd = len(ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for j in range(nd):
                ture_point = true_res_cls[ids[j]]
                point1 = points[j, :]  # [2]
                dis_min = np.inf
                PGT = ture_point['point_pos']  # [num_guidewire, 2]
                if len(PGT) > 0:
                    dis_square = np.square(PGT[:, 0] - point1[0]) + np.square(PGT[:, 1] - point1[1])
                    dis_min = np.min(dis_square)
                if dis_min < threshold * threshold:
                    tp[j] = 1.
                else:
                    fp[j] = 1.

            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / np.maximum(float(npos), np.finfo(np.float64).eps)
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self._voc_ap(rec, prec)
            APs[cls] = ap
        return APs


def makeGaussian(height, width, sigma=3, center=None):
    """ make一个高斯核，是生成heatmap的一个部分
    """
    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)[:, np.newaxis]
    if center is None:
        x0 = width // 2
        y0 = height // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / (sigma ** 2))


class MAPCallbackBox:
    def __init__(self,
                 model,
                 val_dataset,
                 class_names,
                 inference_num=50,
                 batch_size=1):
        super(MAPCallbackBox, self).__init__()
        self.model = model
        self.inference_num = inference_num
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.val_dataset = val_dataset
        self.batch_size = batch_size

    def _voc_ap(self, rec, prec):
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def calculate_result(self):
        true_res = {}
        pred_res = []
        inference_time = 0
        for i in range(self.inference_num):
            image, class_ids, bbox, point = modellib.load_image_gt_eval(self.val_dataset, i)
            start = time.time()
            results = self.model.detect([image])[0]
            end = time.time()
            inference_time = inference_time + (end - start)
            out_boxes = results['rois']
            out_scores = results['scores']
            if len(out_boxes) > 0:
                for out_box, out_score in zip(
                        out_boxes, out_scores):
                    pred_res.append([i, 0, out_score, out_box])
            #                     print([i, 0, out_score, out_box])
            true_res[i] = bbox  # [num_guidewire, 4]
        #             print(bbox)
        print('avg_infer_time:' + str(inference_time / self.inference_num))
        return true_res, pred_res

    def compute_iou(self, box, boxes, box_area, boxes_area):
        # Calculate intersection areas
        y1 = np.maximum(box[0], boxes[:, 0])
        y2 = np.minimum(box[2], boxes[:, 2])
        x1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[3], boxes[:, 3])
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        union = box_area + boxes_area[:] - intersection[:]
        iou = intersection / union
        return iou

    def compute_aps(self, true_res, pred_res):
        APs = {}
        for cls in range(self.num_classes):
            pred_res_cls = [x for x in pred_res if x[1] == cls]
            if len(pred_res_cls) == 0:
                APs[cls] = 0
                continue
            true_res_cls = {}
            npos = 0
            for index in true_res:  # index is the image_id
                guidewires = true_res[index]  # [num_guidewire, 4]
                npos += len(guidewires)  # compute recall
                point_pos = np.array([x for x in guidewires])  # [num_guidewire, 4]
                true_res_cls[index] = {
                    'point_pos': point_pos,
                }
            ids = [x[0] for x in pred_res_cls]
            scores = np.array([x[2] for x in pred_res_cls])
            points = np.array([x[3] for x in pred_res_cls])
            sorted_ind = np.argsort(-scores)
            points = points[sorted_ind, :]  # sorted
            ids = [ids[x] for x in sorted_ind]  # sorted

            nd = len(ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for j in range(nd):
                ture_point = true_res_cls[ids[j]]
                box = points[j, :]  # [4]
                PGT = ture_point['point_pos']  # [num_guidewire, 4]
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                boxes_area = (PGT[:, 2] - PGT[:, 0]) * (PGT[:, 3] - PGT[:, 1])

                if len(PGT) > 0:
                    IOU = self.compute_iou(box, PGT, box_area, boxes_area)
                    iou_max = np.max(IOU)
                if iou_max > 0.5:
                    tp[j] = 1.
                else:
                    fp[j] = 1.

            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / np.maximum(float(npos), np.finfo(np.float64).eps)
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self._voc_ap(rec, prec)
            APs[cls] = ap
        return APs

    def on_epoch_end(self, logs=None):
        logs = logs or {}
        K.set_learning_phase(0)  # For BN
        true_res, pred_res = self.calculate_result()
        APs = self.compute_aps(true_res, pred_res)
        for cls in range(self.num_classes):
            if cls in APs:
                print(self.class_names[cls] + ' ap: ', APs[cls])
        mAP = np.mean([APs[cls] for cls in APs])
        print('mAP: ', mAP)
        logs['mAP'] = mAP


class MAPCallbackPCK:
    def __init__(self,
                 model,
                 val_dataset,
                 class_names,
                 inference_num=50,
                 batch_size=1):
        super(MAPCallbackPCK, self).__init__()
        self.model = model
        self.inference_num = inference_num
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.val_dataset = val_dataset
        self.batch_size = batch_size

    def _voc_ap(self, rec, prec):
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def check_dt(self, box, gtbox):
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (gtbox[:, 2] - gtbox[:, 0]) * (gtbox[:, 3] - gtbox[:, 1])
        IOU = self.compute_iou(box, gtbox, box_area, boxes_area)
        iou_max = np.max(IOU)
        if iou_max > 0.5:
            return True
        else:
            return False

    def calculate_result(self):
        true_res = {}
        pred_res = []
        inference_time = 0
        for i in range(self.inference_num):
            image, class_ids, bbox, point = modellib.load_image_gt_eval(self.val_dataset, i)
            start = time.time()
            out_masks = self.model.localization([image], [bbox])[0]
            # print(out_masks.shape)
            end = time.time()
            inference_time = inference_time + (end - start)
            for out_mask in out_masks:
                det_point = np.unravel_index(out_mask[:, :, 0].argmax(), out_mask[:, :, 0].shape)
                pred_res.append([i, 0, det_point[1] + 1, det_point[0] + 1])
                # print([i, 0, det_point[1] + 1, det_point[0] + 1])
                det_point = np.unravel_index(out_mask[:, :, 1].argmax(), out_mask[:, :, 1].shape)
                pred_res.append([i, 1, det_point[1] + 1, det_point[0] + 1])
                # print([i, 1, det_point[1] + 1, det_point[0] + 1])
            true_res[i] = point  # [num_guidewire, num_point, 2]
        print('avg_infer_time:' + str(inference_time / self.inference_num))
        return true_res, pred_res

    def compute_iou(self, box, boxes, box_area, boxes_area):
        # Calculate intersection areas
        y1 = np.maximum(box[0], boxes[:, 0])
        y2 = np.minimum(box[2], boxes[:, 2])
        x1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[3], boxes[:, 3])
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        union = box_area + boxes_area[:] - intersection[:]
        iou = intersection / union
        return iou

    def compute_pck(self, true_res, pred_res, threshold):
        APs = {}
        for cls in range(self.num_classes):
            true_num = 0
            pred_res_cls = [x for x in pred_res if x[1] == cls]
            num_all = len(pred_res_cls)
            if num_all == 0:
                APs[cls] = 0
                continue
            true_res_cls = {}
            for index in true_res:  # index is the image_id
                guidewires = true_res[index]  # [num_guidewire, num_point, 2]
                point_pos = np.array([x[cls] for x in guidewires])  # [num_guidewire, 2]
                true_res_cls[index] = {
                    'point_pos': point_pos,
                }

            for j in pred_res_cls:
                ture_point = true_res_cls[j[0]]
                point1 = j[2:]  # [2]
                PGT = ture_point['point_pos']  # [num_guidewire, 2]
                if len(PGT) > 0:
                    dis_square = np.square(PGT[:, 0] - point1[0]) + np.square(PGT[:, 1] - point1[1])
                    dis_min = np.min(dis_square)
                if dis_min < threshold * threshold:
                    true_num += 1
            print(true_num, num_all)
            APs[cls] = true_num / num_all
        return APs

    def on_epoch_end(self, logs=None):
        logs = logs or {}
        K.set_learning_phase(0)  # For BN
        true_res, pred_res = self.calculate_result()
        for th in [3, 5, 7, 9]:
            APs = self.compute_pck(true_res, pred_res, th)
            for cls in range(self.num_classes):
                if cls in APs:
                    print(self.class_names[cls] + ' ap: ', APs[cls])
            mAP = np.mean([APs[cls] for cls in APs])
            print('mAP: ', mAP)
            logs['mAP'] = mAP


def read_point(txt_path):
    with open(txt_path, 'r')as f:
        string = f.readlines()
        num_guidewire = len(string)
        point = np.zeros([num_guidewire, 2, 2], dtype=np.int32)
        bbox = np.zeros([num_guidewire, 4], dtype=np.int32)
        for index, s in enumerate(string):
            item = [int(i) for i in s[:-1].split(' ')]
            bbox[index] = np.array([item[2], item[0], item[3], item[1]])
            point[index, 0] = np.array([item[4], item[5]], dtype=np.int32)
            point[index, 1] = np.array([item[6], item[7]], dtype=np.int32)
    return bbox, point


def make_output(filename, model, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_list = os.listdir(filename)
    for pic_name in image_list:
        image_path = os.path.join(filename, pic_name)
        image = cv2.imread(image_path)
        results = model.detect([image])[0]
        out_boxes = results['rois']
        out_scores = results['scores']
        out_masks = results['masks']
        if len(out_boxes) > 0:
            for out_box, out_score, out_mask in zip(
                    out_boxes, out_scores, out_masks):
                image = cv2.rectangle(image, (out_box[1], out_box[0]), (out_box[3], out_box[2]), (255, 0, 0), 2)
                det_point = np.unravel_index(out_mask[:, :, 0].argmax(), out_mask[:, :, 0].shape)
                image = cv2.circle(image, (det_point[1] + 1, det_point[0] + 1), 8, (0, 255, 0), 4)
                # pred_res.append([i, 0, det_point[1] + 1, det_point[0] + 1])
                det_point = np.unravel_index(out_mask[:, :, 1].argmax(), out_mask[:, :, 1].shape)
                image = cv2.circle(image, (det_point[1] + 1, det_point[0] + 1), 8, (0, 0, 255), 4)
                # pred_res.append([i, 1, det_point[1] + 1, det_point[0] + 1])
        out_path = os.path.join(output_dir, pic_name[:-4]+'.png')
        cv2.imwrite(out_path, image)



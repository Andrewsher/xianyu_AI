from keras.losses import binary_crossentropy
import keras.backend as K
import tensorflow as tf

from segmentation_models.losses import jaccard_loss
from segmentation_models.metrics import iou_score

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1 ) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + 1)

# def iou_score(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (intersection + 1) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + 1 - intersection)

def dice_0(y_true, y_pred):
    return dice_coef(y_true[:, :, :, 0], y_pred[:, :, :, 0])

def dice_1(y_true, y_pred):
    return dice_coef(y_true[:, :, :, 1], y_pred[:, :, :, 1])

def dice_2(y_true, y_pred):
    return dice_coef(y_true[:, :, :, 2], y_pred[:, :, :, 2])

def dice_3(y_true, y_pred):
    return dice_coef(y_true[:, :, :, 3], y_pred[:, :, :, 3])

def dice_loss(y_true, y_pred):
    return 1 - (dice_0(y_true, y_pred) + dice_1(y_true, y_pred) + dice_2(y_true, y_pred) + dice_3(y_true,y_pred)) / 4.

def iou_0(y_true, y_pred, per_image=True):
    return iou_score(y_true[:, :, :, 0], y_pred[:, :, :, 0], per_image=per_image)

def iou_1(y_true, y_pred, per_image=True):
    return iou_score(y_true[:, :, :, 1], y_pred[:, :, :, 1], per_image=per_image)

def iou_2(y_true, y_pred, per_image=True):
    return iou_score(y_true[:, :, :, 2], y_pred[:, :, :, 2], per_image=per_image)

def iou_3(y_true, y_pred, per_image=True):
    return iou_score(y_true[:, :, :, 3], y_pred[:, :, :, 3], per_image=per_image)

def iou_loss_3_sum(y_true, y_pred):
    return 3 - (iou_1(y_true, y_pred, per_image=False) + iou_2(y_true, y_pred, per_image=False) + iou_3(y_true, y_pred, per_image=False))

def iou_3_sum(y_true, y_pred):
    return (iou_1(y_true, y_pred) + iou_2(y_true, y_pred) + iou_3(y_true, y_pred)) / 3

def iou_loss_weight_sum(y_true, y_pred):
    return 6 - (0.3*iou_0(y_true, y_pred) + iou_1(y_true, y_pred) + iou_2(y_true, y_pred) + iou_3(y_true, y_pred))

def dice_iou_loss(y_true, y_pred):
    return jaccard_loss(y_true, y_pred) + dice_loss(y_true, y_pred)

# focal loss with multi label
def focal_loss(classes_num, gamma=2., alpha=.25, e=0.1):
    # classes_num contains sample number of each classes
    def focal_loss_fixed(target_tensor, prediction_tensor):
        '''
        prediction_tensor is the output tensor with shape [None, 100], where 100 is the number of classes
        target_tensor is the label tensor, same shape as predcition_tensor
        '''
        import tensorflow as tf
        from tensorflow.python.ops import array_ops
        from keras import backend as K

        #1# get focal loss with no balanced weight which presented in paper function (4)
        zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
        one_minus_p = array_ops.where(tf.greater(target_tensor,zeros), target_tensor - prediction_tensor, zeros)
        FT = -1 * (one_minus_p ** gamma) * tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0))

        #2# get balanced weight alpha
        classes_weight = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)

        total_num = float(sum(classes_num))
        classes_w_t1 = [ total_num / ff for ff in classes_num ]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ ff/sum_ for ff in classes_w_t1 ]   #scale
        classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=prediction_tensor.dtype)
        classes_weight += classes_w_tensor

        alpha = array_ops.where(tf.greater(target_tensor, zeros), classes_weight, zeros)

        #3# get balanced focal loss
        balanced_fl = alpha * FT
        balanced_fl = tf.reduce_mean(balanced_fl)

        #4# add other op to prevent overfit
        # reference : https://spaces.ac.cn/archives/4493
        nb_classes = len(classes_num)
        fianal_loss = (1-e) * balanced_fl + e * K.categorical_crossentropy(K.ones_like(prediction_tensor)/nb_classes, prediction_tensor)

        return fianal_loss
    return focal_loss_fixed



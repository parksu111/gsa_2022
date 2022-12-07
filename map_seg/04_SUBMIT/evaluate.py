import sys
import pandas as pd
import numpy as np
import time

max_pixel = 512*512

def cal_iou(answer, predicted, class_number_answer, class_number_pred):

    """
    :param answer: array of numbers (ex. "0 4 10 5 22 7") for one id
    :param predicted: array of numbers (ex. "0 4 10 5 22 7") for one id
    :return: (float) iou
    """
    assert len(predicted)%2 == 0, 'Wrong prediction format: contains odd number of values'
    assert (predicted[-1] + predicted[-2] -1) < max_pixel, "Maximum pixel index exceeds maximum image size (512*512)"

    if class_number_pred != class_number_answer:
        return 0

    answer_mask = []
    for i in range(len(answer)//2):
        answer_mask.extend(list(range(answer[i*2], answer[i*2]+answer[i*2+1])))

    pred_mask = []
    for i in range(len(predicted) // 2):
        pred_mask.extend(list(range(predicted[i * 2], predicted[i * 2] + predicted[i * 2 + 1])))

    answer_mask = np.array(answer_mask)
    pred_mask = np.array(pred_mask)

    intersection = np.intersect1d(answer_mask,pred_mask)
    union = np.union1d(answer_mask,pred_mask)
    iou = len(intersection)/len(union) 
    return iou


def load_result(gt_df, pred_df):

    gt_cols = list(gt_df.columns)
    gt_cols.remove('public')
    assert set(gt_cols)==set(pred_df.columns), 'Column names of prediction and answer are not the same'
    assert len(gt_df)==len(pred_df), 'The number of predictions and answers are not the same'
    assert set(gt_df['img_id'])==set(pred_df['img_id']), 'Prediction is missing or contains extra file_name'
    assert gt_df['img_id'].tolist() == pred_df['img_id'].tolist(), 'file_name should be ordered as the sample submission'
    
    class_dic = {"building":1}

    
    assert len(set(pred_df['class'].unique()).union(set(class_dic.keys())))==1, "Invalid class type included." 

    assert sum(pred_df['prediction'].isna())==0, "Either an empty or an invalid value exists"
    

    gt_classes = gt_df['class'].map(lambda x : class_dic[x]).tolist()
    pred_classes =  pred_df['class'].map(lambda x : class_dic[x]).tolist()

    
    pred_list = [list(map(int, pred.split(" "))) for pred in pred_df['prediction'].tolist()]
   
    gt_list = [list(map(int, gt.split(" "))) for gt in gt_df['prediction'].tolist()]

    ious = []
    for pred, gt, gt_c, pred_c, in zip(pred_list, gt_list, gt_classes, pred_classes):
        iou = cal_iou(gt, pred, gt_c, pred_c)
        ious.append(iou)

    return np.array(gt_classes), np.array(ious), gt_df['public'].values

def mIoU(answer_path, pred_path):

    classes, ious, p_type_li = load_result(pd.read_csv(answer_path), pd.read_csv(pred_path))
    pub_iou_per_class = np.zeros(1)
    prv_iou_per_class = np.zeros(1)

    for c in range(1):
        pub_classes = classes[p_type_li]
        pub_ious = ious[p_type_li]
        indexes = np.where(pub_classes == c+1)[0]
        matched_ious = pub_ious[indexes]
        pub_iou_per_class[c] = matched_ious.mean()

    for c in range(1):
        prv_idx = np.logical_not(p_type_li)
        prv_classes = classes[prv_idx]
        prv_ious = ious[prv_idx]
        indexes = np.where(prv_classes == c + 1)[0]
        matched_ious = prv_ious[indexes]
        prv_iou_per_class[c] = matched_ious.mean()

    score = pub_iou_per_class.mean()
    pScore = prv_iou_per_class.mean()

    return score, pScore


if __name__ == '__main__':
    answer = sys.argv[1]
    pred = sys.argv[2]

    try:
        import time

        start = time.time()
        score, pScore = mIoU(answer, pred)
        print(f'score={score},pScore={pScore}')
        print(f'Elapsed Time: {time.time() - start}')
    except Exception as e:
        print(f'evaluation exception error: {e}', file=sys.stderr)
        sys.exit()


import sys
import pandas as pd
from sklearn.metrics import f1_score

def evaluation_metric(label, prediction):
    return f1_score(label, prediction, average='macro')

def evaluate(label, prediction):
    label = list(map(int, label))
    prediction = list(map(int, prediction))
        
    return evaluation_metric(label=label, prediction=prediction)

def load_result(path, pred=False, cols=None):
    label_encoding = {'W':0, 'N1':1, 'N2':2, 'N3':3, 'R':4}
    try:
        result = pd.read_csv(path)
        
        if pred:
            assert set(result.columns)==set(cols), 'Column names of prediction and answer are not the same'
        
        for i,sample in enumerate(result['stage']):
            assert sample in label_encoding.keys(), f'invalid category: {sample}'
            result.loc[i, 'stage'] = label_encoding[sample]
            
        if pred == False: #answer
            p_type_li = list(result['public'])
            cols = list(result.columns)
            cols.remove('public')
        else:
            p_type_li = None
    
        return list(result['rec_id']), list(result['stage']), p_type_li, cols
    
    except Exception as e:
        assert False, e

def F1Score(answer_path, pred_path):
    a_id, answer, p_type_li, a_cols = load_result(answer_path)
    p_id, pred, _, _ = load_result(pred_path, pred=True, cols=a_cols)
    
    assert len(a_id) == len(p_id), 'The number of predictions and answers are not the same'
    assert set(p_id) == set(a_id), 'Prediction file is missing ids or contains extra ids'
    assert a_id == p_id, 'The prediction ids should be ordered as the sample submission file'
    
    pub_a_id, pub_answer, prv_a_id, prv_answer = [],[],[],[]
    pub_p_id, pub_pred, prv_p_id, prv_pred = [],[],[],[]
    for idx,t in enumerate(p_type_li):
        if t:
            pub_a_id.append(a_id[idx])
            pub_answer.append(answer[idx])
            pub_p_id.append(p_id[idx])
            pub_pred.append(pred[idx])
        else:
            prv_a_id.append(a_id[idx])
            prv_answer.append(answer[idx])
            prv_p_id.append(p_id[idx])
            prv_pred.append(pred[idx])
    
    # sort
    pub_ans = pd.DataFrame({'id': pub_a_id, 'answer':pub_answer}).sort_values('id',ignore_index=True)
    prv_ans = pd.DataFrame({'id': prv_a_id, 'answer':prv_answer}).sort_values('id',ignore_index=True)
    pub_pred = pd.DataFrame({'id': pub_p_id, 'answer': pub_pred}).sort_values('id',ignore_index=True)
    prv_pred = pd.DataFrame({'id': prv_p_id, 'answer': prv_pred}).sort_values('id',ignore_index=True)
    
    #f1 score
    score = evaluate(label = pub_ans['answer'], prediction=pub_pred['answer'])
    pScore = evaluate(label = prv_ans['answer'], prediction=prv_pred['answer'])
    
    return score, pScore

if __name__ == '__main__':
    answer = sys.argv[1]
    pred = sys.argv[2]
    
    try:
        import time
        start = time.time()
        score, pScore = F1Score(answer, pred)
        print(f'score={score},pScore={pScore}')
        print(f'Elapsed Time: {time.time() - start}')
        
    except Exception as e:
        print(f'evaluation exception error: {e}', file=sys.stderr)
        sys.exit()
import pandas as pd
from scipy.stats import pearsonr

dimensions = [
    # 'conciseness', 
    # 'fluency', 
    # 'clarity', 
    # 'relevance', 
    # 'consistency', 
    'answerability', 
    # 'answer_consistency'
]

data_dir = '../../result/llama3/vanilla/'
model = 'llama3'

for dim in dimensions:
    df = pd.read_excel(data_dir+'qgeval-{}.xlsx'.format(dim))
    col = dim+'_'+model+'_score'
    scores = [float(x) for x in df[dim].to_list()]
    preds = [float(x)for x in df[col].to_list()]
    assert len(scores) == len(preds)
    list1, list2 = [], []
    for s, p in zip(scores, preds):
        if p == 999:
            continue
        list1.append(s)
        list2.append(p)
    correlation_coefficient, p_value = pearsonr(list1, list2)
    print('Dimension {}: data_count={}, pearson_corr={}'.format(
        dim, len(list1), round(correlation_coefficient, 3)
    ))

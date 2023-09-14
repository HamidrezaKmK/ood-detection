
def get_roc_curves(df, metric_A, metric_B):
    """
    Given a dataframe, return a dictionary of 2d ROC AUCs
    
    'generated-vs-test': val,
    'generated-vs-ood': val,
    'train-vs-test': val,
    'train-vs-ood': val,
    
    """
    
    ret = {
        'generated-vs-test': None,
        'generated-vs-ood': None,
        'train-vs-test': None,
        'train-vs-ood': None,
    }
    for key in ret.keys():
        first, second = key.split("-vs-")
        # check if df contains both first and second in the name column
        if len(df[df["name"].str.startswith(first)]) == 0 or len(df[df["name"].str.startswith(second)]) == 0:
            continue
        
        df_first = df[df["name"].str.startswith(first)]
        df_second = df[df["name"].str.startswith(second)]
        
        pos_x = df_first[metric_A].to_numpy()
        pos_y = df_first[metric_B].to_numpy()
        neg_x = df_second[metric_A].to_numpy()
        neg_y = df_second[metric_B].to_numpy()
        
        auc_curves = get_2d_roc_curve(pos_x, pos_y, neg_x, neg_y)
        ret[key] = auc_curves
    
    return ret

def get_2d_roc_curve(
    pos_x : np.ndarray,
    pos_y : np.ndarray,
    neg_x : np.ndarray,
    neg_y : np.ndarray,
):
    N = len(pos_x)
    
    all_x = np.concatenate([pos_x, neg_x, np.array([np.min(pos_x) + 1e-6, np.max(neg_x) + 1e-6])])
    all_y = np.concatenate([pos_y, neg_y, np.array([np.min(pos_y) + 1e-6, np.max(neg_y) + 1e-6])])
    all_fps = {}
    
    for x in all_x:
        for y in all_y:
            # the classifier is >x and >y
            tp = np.sum((pos_x >= x) & (pos_y >= y))
            fp = np.sum((neg_x >= x) & (neg_y >= y))
            if fp not in all_fps:
                all_fps[fp] = []
            all_fps[fp].append(tp)
    
    ret_tp = []
    ret_fp = []
    for fp in sorted(all_fps.keys()):
        ret_fp.append(1.0 * fp / N)
        ret_tp.append(1.0 * max(all_fps[fp]) / N)
        
    return np.array(ret_fp), np.array(ret_tp)

def get_auc(curve_x, curve_y):
    """
    Given a curve, return the area under the curve
    """
    auc = 0.0
    for i in range(1, len(curve_x)):
        auc += (curve_x[i] - curve_x[i - 1]) * (curve_y[i] + curve_y[i - 1]) / 2.0
    return auc
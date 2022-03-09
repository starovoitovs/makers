import logging
from datetime import datetime

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, r2_score

logger = logging.getLogger(__name__)


def get_utc_timestamp():
    return datetime.utcnow().strftime('%Y%m%d%H%M%S')


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def get_lob_columns(depth):
    return [x for i in range(depth) for x in [f'bp{i}', f'bv{i}', f'ap{i}', f'av{i}']]


def prepare(df, idxnum, depth, window_size):

    idx = [np.arange(x - window_size + 1, x + 1) for x in idxnum]

    lob_columns = get_lob_columns(depth)
    X1 = df[lob_columns].to_numpy()[idx]

    # subtract midprice
    bid, ask = X1[:, -1, [0, 2]].T
    mid_price = (bid + ask) / 2
    pmask = [x for x in np.arange(4 * depth) if x % 2 == 0]
    X1[:, :, pmask] -= mid_price[:, np.newaxis, np.newaxis]

    # add axis
    X1 = X1[:, :, :, np.newaxis]

    return X1


def get_metrics(y_true, y_pred):

    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred)
    mat = confusion_matrix(y_true, y_pred)
    mat_str = '\n'.join([' ' * 20 + ' '.join(['{:7}'.format(x) for x in row]) for row in mat])
    r2 = r2_score(y_true, y_pred)

    output = (
        f"{'r2':20}{r2:7.2f}\n"
        f"{'precision':20}{' '.join(['{:7.4f}'.format(x) for x in precision])}\n"
        f"{'recall':20}{' '.join(['{:7.4f}'.format(x) for x in recall])}\n"
        f"{'fscore':20}{' '.join(['{:7.4f}'.format(x) for x in fscore])}\n"
        f"{'support':20}{' '.join(['{:7}'.format(x) for x in support])}\n\n"
        f"{'confusion matrix':20}\n{mat_str}\n"
    )

    return output

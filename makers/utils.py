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

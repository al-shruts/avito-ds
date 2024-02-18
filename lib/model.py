import pandas as pd

from typing import Tuple, Union
from transformers import pipeline


def task1(description: list):
    classifier = pipeline("text-classification", model="./classification/", tokenizer="./classification/")

    predict = classifier(description)
    predict_df = pd.DataFrame(predict)

    return predict_df.apply(lambda row: 1 - row['score'] if row['label'] == 'LABEL_0' else row['score'], axis=1)


def task2(description: str) -> Union[Tuple[int, int], Tuple[None, None]]:
    description_size = len(description)
    if description_size % 2 == 0:
        return None, None
    else:
        return 0, description_size

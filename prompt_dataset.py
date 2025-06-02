"""Dataset loading utilities."""
import functools
import pandas as pd
import numpy as np


@functools.lru_cache(maxsize=1)
def load_dataset(dataset_path):
  return pd.read_pickle(dataset_path)


def iterator(dataset_path="/content/drive/MyDrive/DRaFT/generated_data/dataset.pkl",
             return_tuples=False, group_col="id", n_epochs=1, seed=42):
  df = load_dataset(dataset_path)
  rng = np.random.RandomState(seed)
  groups = list(df.groupby(group_col))
  for _ in range(n_epochs):
      rng.shuffle(groups)
      for _, rows in groups:
          id = rows['id'].iloc[0]
          description = rows['description'].iloc[0]
          questions = rows['question'].tolist()
          choices = rows['choices'].tolist()
          answers = rows['answer'].tolist()
          if return_tuples:
            yield (id, description, questions, choices, answers)
          else:
            yield rows

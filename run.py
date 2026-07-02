import pandas as pd
from tests.mask import maskcolumns

masked, truth = maskcolumns(pd.read_csv('tests/data/test.csv'), exclude=['Student_ID'], ratio=0.15, seed=205)
masked.to_csv('masked.csv', index=False)
truth.to_csv('truth.csv', index=False)
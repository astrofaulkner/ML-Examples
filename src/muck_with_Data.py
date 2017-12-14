import pylab as pyl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def run_test():
    
    # Read in mock data
    df = pd.read_csv('MockData/MOCK_DATA.csv')
    #df.info()

    # Add additional column with binary value (0, 1) to test classification
    # 0: Pass; 1: Fail
    df['result'] = np.random.randint(0, 2, df.shape[0])
    #print(df.head())

    # Create histograms
    df['result'].value_counts().plot(kind='bar', title='Results')

    df['ext'] = 0
    for i in range(len(df.index)):
        df['ext'][i] = df['name'][i].split('.')[1]
    #print(df.head())

    df2 = df.loc[df['ext'].isin(['js', 'pcl'])]
    pclass_xt = pd.crosstab(df2['ext'], df2['result'])
    pclass_xt_pct = pclass_xt.div(pclass_xt.sum(1).astype(float), axis=0)
    pclass_xt_pct.plot(kind='bar', stacked=True, title='Extension vs. Result')
    #pyl.xlabel('Extension')
    #pyl.ylabel('Result')

    pyl.show()


if __name__ == "__main__":
    run_test()

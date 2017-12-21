import pylab as pyl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import math

def run_test():
    
    # Read in mock data
    df = pd.read_csv('MockData/MOCK_DATA.csv')
    #df.info()

    # Add additional column with binary value (0, 1) to test classification
    # 0: Pass; 1: Fail
    df['result'] = np.random.randint(0, 2, df.shape[0])
    #print(df.head())

    # Create histograms
    #df['result'].value_counts().plot(kind='bar', title='Results')

    # Create 'ext' column for file extensions
    df['ext'] = 0
    for i in range(len(df.index)):
        df['ext'][i] = df['name'][i].split('.')[1]
    #print(df.head())

    # Plot cross table of various extensions vs results
    df2 = df.loc[df['ext'].isin(['js', 'pcl'])]
    pclass_xt = pd.crosstab(df2['ext'], df2['result'])
    pclass_xt_pct = pclass_xt.div(pclass_xt.sum(1).astype(float), axis=0)
    #pclass_xt_pct.plot(kind='bar', stacked=True, title='Extension vs. Result')

    # Generate a mapping between file extensions and integers
    exts = sorted(df['ext'].unique())
    exts_map = dict(zip(exts, range(0, len(exts) + 1)))
    df['ext_val'] = df['ext'].map(exts_map).astype(int)
    #print(df.head())

    # Plot the ext_val histogram
    #df['ext_val'].hist()
    #pyl.title('Ext. Mapping Histogram')

    # Plot cross table of all extensions vs result
    pclass_xt = pd.crosstab(df['result'], df['ext_val'])
    pclass_xt_pct = pclass_xt.div(pclass_xt.sum(1).astype(float), axis=0)
    #pclass_xt_pct.plot(kind='bar', stacked=True, title='Mapped Extension vs. Result')

    # Plot joint plot between ext_val and len(filename)
    df['FN_len'] = 0
    for i in range(len(df.index)):
        df['FN_len'][i] = len(df['name'][i])-len(df['ext'][i])
    #sb.jointplot(data=df, x='ext_val', y='FN_len', kind='reg', color='g')
    sb.kdeplot(df.ext_val, df.FN_len)

    # Create pivot table of result vs ext for mean filename length
    df['ext_val_trumpd'] = 0
    for i in range(len(df.index)):
        df['ext_val_trumpd'][i] = df['ext_val'][i] % 10.
    df3 = df.pivot_table(index='result', columns='ext_val_trumpd', values='FN_len', aggfunc=np.median)
    #sb.heatmap(df3, annot=True, fmt=".1f")

    # Create Pearson's correlation coefficient matrix for dataframe
    #sb.heatmap(df.corr(), annot=True, fmt=".2f")

    pyl.show()


if __name__ == "__main__":
    run_test()

# ML-Examples
# Will contain various source code repos and notes.
# Notes here are intentionally vague.

For our use case, we must first define the problem question - what is it we need to know
and how do we measure success?

(1) - We need to predict a label outcome, based on feature sets.
    -- Classification: Of value / No Value
    - We are unsure if feature sets contain correlations/associations with value.
    -- Features consist of all data (DB) columns.
    - Quality Assurance: What fraction of data was looked at and evaluated?
    -- We need to beat the coin toss at initial ingest of new data.
    -- We need to beat the success rate of previous (ad-hoc) methods.
    -- Initial success will be measured by fraction of correctly labeled data. (> 50%)
    - Is the data able to address the problem?
    -- Analyze by N-1 feature sets, use data feedback reports to label training data, and
       only train on data previously analyzed and reported (test on the rest).

(2) - We need to classify based on data set.
    - Classification/Clustering: Purpose? / Author genre?
    - Quality Assurance: What categories were determined in the data?
    -- We need to consider data host type, content, and frequency of generation.
    -- Success will be measured in fraction of correctly labeled data. (> 90%)
    -- May not have a training data set!

_____________________________________________________________________________________________

CLEAN THE DATA!

- Never clean data by hand. Always use scripts/code to clean/manage the selected data to
  maintain reproducability and generate a record for peer-review.
- One good way to maintain situational awareness (SA) with test/training data quality is
  to include Assertions in Python to halt processing in failed instances.
$ assert 1 == 2 # will cause program to halt execution
$ assert date > 01/01/2000 # will halt execution if old dates are entered

Seaborn Python module can produce instant NxN pair correlation plots with Pandas DataFrames!
$ import seaborn as sb
$ sb.pairplot(dataframe.dropna()[, hue='class']) # 'hue' option if color code data

- What this means is you can instantly (relatively quickly) isolate any erraneous data, or
  significant outliers. These points are either bad, need to be removed, or well understood.

_____________________________________________________________________________________________

STUDY THE DATA!

Seaborn violin plots are box plots with data density taken into account.
$ plt.figure(figsize=(10,10))
$ for column_index, column in enumerate(dataframe.columns):
$    if column == 'class': # 'class' is the column you wish to compare against
$       continue
$    plt.subplot(n, n, column_index + 1) # 'n' forms a nxn plot of column features
$    sb.violinplot(x='class', y=column, data=dataframe)

- Check if data is normally distributed; otherwise, cannot use models that assume the underlying
  data is normally distributed.

Pandas DataFrames has built-in functions to display gereralized information about the data.
$ df.describe() # provides matrix of: count, mean, std, min, etc. of dataframe columns
$ df.info() # provides list of columns, data types, and number of non-null entries in each column
$ df.dtypes # provides list of data types for each column only

View distributions in MatPlotLib:
$ import pylab as plt
$ plt.rc('figure', figsize=(10, 5)) # set default size of MatPlotLib figures
$ figsize_with_subplots = (10, 10) # size of figures with subplots
$ bin_size = 10 # size of histogram bins
$ fig = plt.figure(figsize=figsize_with_subplots)
$ fig_dims = (3, 2)
$ plt.subplot2grid(fig_dims, (0, 0))
$ df['column name'].value_counts().plot(kind='bar', title='Subplot Title') # great way to plot histos
$ plt.subplot2grid(fig_dims, (0, 1))
$ df['column name'].hist()
$ plt.title('Subplot Title')

Get a cross table of different columns in dataframe:
$ pclass_xt = pd.crosstab(df['column_1'], df['column_2'])

Plot histogram of cross table above (previous step):
$ pclass_xt_pct = pclass_xt.div(pclass_xt.sum(1).astype(float), axis=0) # normalize to 1.0
$ pclass_xt_pct.plot(kind='bar', stacked=True, title='Some Title')
$ plt.xlabel('column_1')
$ plt.ylabel('column_2')

_____________________________________________________________________________________________

Model Accuracy:

- One can train on the entire data set and then test against that same data set.
-- This would yield 'training accuracy'. Measured as fraction of correctly labeled test data
   over all test data (remember that test data = all data).
-- Downside is you tend to overtrain the model.  Better training accuracy can mean less
   generalized model to use against out-of-sample data.

- One can split the entire data set into a training set and a testing set. In some circles
  this can be referred to as 'blinding'. You 'blind' your model from the testing subset of data
  and only train on a small, separate subset (training set).
-- I have previously exercised training on 10% of the entire data set (blinded from 90%).
-- This would yield 'testing accuracy'. Measured as fraction of correctly labeled test data
   over all test data (remember that test data is separate subset of entire data set).
-- Downside is a high-variance estimate of out-of-sample accuracy (model may be too generalized).

- One could implement K-fold cross-validation to marry benefits of generalization and
  acceptable level of specificity. You split the data set into K subsets, choose one subset
  to be the training data and the rest as the test data. Measure the test accuracy, and then
  repeat process with choosing a different subset as the training data. Repeat until all
  K subsets have been used separately as training sets. Average the testing accuracies.

- Confusion matrix is used to score your model: Binary Matrix (example) is as follows:

  -------------------------------------------------------
  |                  | Condition True | Condition False |
  |------------------|----------------|-----------------|
  | Prediction True  | True Positive  | False Positive  |
  |------------------|----------------|-----------------|
  | Prediction False | False Negative | True Negative   |
  -------------------------------------------------------

  Precision = True Positive (TP) / ( True Positive (TP) + False Positive (FP) )
  Recall = TP / ( TP + False Negative (FN) )
  F1 = 2 * TP / (2 * TP + FP + FN )  # model is best when F1 = 1.0 and worst at F1 = 0.0

-- F-measures do not account for True Negatives. Possible alternatives include:
   Matthews Correlation Coefficient, Informedness, or Cohen's Kappa.

____________________________________________________________________________________________

Example data:

- https://digitalcorpora.org provides example digital forensic data.
- www.mockaroo.com generates random data.

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

____________________________________________________________________________________________

Example data:

- https://digitalcorpora.org provides example digital forensic data.
- www.mockaroo.com generates random data.

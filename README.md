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
    -- Analyze by N-1 feature sets, use data reports to label training data, and only train
       on data previously analyzed and reported (test on the rest).

(2) - We need to classify based on data set.
    - Classification/Clustering
    -- Purpose?
    -- Author genre?
    - Quality Assurance: What categories were determined in the data?
    -- We need to consider data host type, content, and frequency of generation.
    -- Success will be measured in fraction of correctly labeled data. (> 90%)
    -- May not have a training data set!

_____________________________________________________________________________________________

CLEAN THE DATA!

- Never clean data by hand. Always use scripts/code to clean/manage the selected data to
  maintain reproducability and generate a record for peer-review.
- One good way to maintain SA with test/training data quality is to include Assertions
  in Python to halt processing in failed instances.
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

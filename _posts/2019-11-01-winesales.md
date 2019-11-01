---
title: "Machine Learning: Wine Sales Prediction Project"
date: 2019-11-01
tags: [machine learning, regression, xgboost]
header:
  image: /assets/images/winesales/header.jpg
  teaser: /assets/images/winesales/teaser.jpg
excerpt: "prediction model comparison using wine properties"
mathjax: "true"
---
Wine is as subjective a product as any with its various flavors, grapes, producers and production methods. What makes a particular wine taste good to one person may easily be the cause of distaste for another. Here I explore the predictability of wine sales based upon various wine properties, ranging from acidity to label appeal to alcohol content. Fourteen (14) different features are explored and then integrated into several different modeling methods, including regression, hurdle models, and gradient boosting.

All code can be found on my [github](https://github.com/joe-cipolla/wine_sales_prediction).


## Data Exploration
This dataset includes over 12,000 records of wine sales, each with its own properties as shown in the table below. The theoretical effect of most of the variables is unknown.  Whether or not the chemical composition of a wine (like sulphates/chlorides content) drives a wine's appeal to customers will be explored. However, variables related to experts’ reviews of the wines should have a positive effect on how many cases are sold, as should ratings on the appeal of the wine bottle label increase the number of sales.

The prediction target is a counting variable, ranging from 0 to 8, equal to the number of cases of wine sold. A variable of this kind necessitates different modeling techniques because of its [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution), as opposed to a binomial or normal. This will require a few transformations to prepare the data for our various modeling techniques.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/winesales/variable_table.jpg" alt="wine variable table">

The pair plots below show how the sale counts (TARGET) were distributed across Volatile Acidity, Label Appeal, and the Normalized Alcohol content. The wine sales distributions across Volatile Acidity and Norm_Alcohol are very representative of how the target variable was distributed across most of the “chemical” variables. The relationship between a wine's chemical composition and its sale count appears to be stochastic (random). However, the Label Appeal variable shows a clear positive correlation with the TARGET variable (top row, 2nd plot from the right).

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/winesales/pairplot_acid_label_ph.png" alt="wine pair plots 1">
<br/><br/>
<img src="{{ site.url }}{{ site.baseurl }}/assets/images/winesales/pairplot_acid_sugar_chlor.png" alt="wine pair plots 2">
<br/><br/>
<img src="{{ site.url }}{{ site.baseurl }}/assets/images/winesales/pairplot_stars_label_acid.png" alt="wine pair plots 3">
<br/><br/>
<img src="{{ site.url }}{{ site.baseurl }}/assets/images/winesales/pairplot_sulf_stars_dens.png" alt="wine pair plots 4">
(In order to have visually distinguishable plotted values, random samplings of 500 of the 12,000 records were gathered for the pair plots. The sampling was run several times to ensure these plots are true reflections of the entire dataset.)
<br/><br/>
<img src="{{ site.url }}{{ site.baseurl }}/assets/images/winesales/Histogram__AcidIndex_byFlag.png" alt="acid index histogram">

One of the “chemistry” variables did prove to be predictive in several of the models that were developed: **AcidIndex**. This is a metric proprietary to the data provider that classifies the amount of acid present in the wine into a range of integers (0-17). The histogram above shows how the proportion of wine sold versus not sold (0 = no sales, 1 = at least one sale) is much higher for AcidIndex values between 6 and 8. Whereas, when the AcidIndex gets above 9, the ratio inverts, and AcidIndex values this high become predictive of the wine not selling any cases at all.

### STARS
One of the most predictive variables across all of the variables was STARS. This is the reputable critic reviews rating, indicating extremely good quality wine for higher values, and poor quality wine for lower values. Unfortunately, STARS also had one of the highest number of missing values, 3,359 of 12,795 (more than 25%).

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/winesales/Histogram__STARS.png" alt="STARS histogram">

The histograms above are colored with the original TARGET variable values, equal to the number of wine cases sold. As would be expected, poor average ratings/reviews resulted in wines that sell less. Wine with only a 1 star rating, frequently did not sell any cases. And the best selling wines (6, 7 and 8 cases sold) usually received more than 3 star ratings. The STARS variable alone is not sufficient to predict the exact number of cases sold, but it did prove useful in first classifying whether or not a wine would sell any cases.

The next step in the project was to prepare the data for the various modeling techniques. Because of the predictive importance of the STARS variable, one of the most important data transformations that took place was the imputation of missing STARS values. Not only did the high proportion of TARGET = 0 observations occur with STARS = 1, these non-selling wines also made up a larger proportion of the missing STARS values. Due to this, the missing STARS values were imputed using 0, to represent no rating at all.

<br/><br/>
## Data Prep
In order to have a clean dataset that could be ingested into the various modeling methods, a few of the variables had to undergo some transformations, including imputation. This is the process of filling in (imputing) missing values with replacement values, subject to the logic of the modeler. Most of the modeling techniques experimented within this report either fail when presented with missing data, or they exclude the entire observation from the model if there are any variables with missing values.

### Imputation
Eight of the fourteen predictor variables found in the original dataset contained missing values, with the majority missing around 600 of the 12,795 values. The Sulphates variable was missing 1,210, and STARS was missing 3,359. Because most of the missing values were found in the chemical measurement variables, and these variables were anticipated to add little predictive power to the models, laborious approaches to missing value imputation (like decision trees) were not deployed in order to optimize time and focus on the model fitting and testing process.

However, in order to give each model a high amount of visibility into the quality of the data and missing values, flags were created for each of the variables that were imputed. For example, a binary (0 or 1) flag was created for Chlorides, with a 0 indicating that that observation did not previously contain a missing value, and a 1 indicating that it did. The majority of the variables containing missing values, especially chemical measurements, did not appear to have a clear relationship with the target variable, so mean imputation was used. This means that each missing value was filled in with the mean for that variable.

The only exception to this was the STARS variable, which was imputed with a 0, as the most likely reason the STARS variable was missing a value was thought to be that the wine actually did not receive a rating at all. Because of the large amount of wines that had missing STARS values as well as had a 0 TARGET value, it seemed appropriate to delineate the STARS as a 0 rather than the mean, to reinforce the predictive effect in the model. The new imputed variables are signified by a leading “IMP_” before the variable name.

<br/><br/>
<img src="{{ site.url }}{{ site.baseurl }}/assets/images/winesales/grape_stomping.jpg" alt="grape stompin'">
### Transformation
One of the main reasons we transform our data before modeling is to make sure the data adheres the various models' assumptions. For example, linear regression assumes that all of the differences between predictions and actual values (referred to as errors) are homoscedastic, or randomly distributed with no relationship to either the target or predictor variables. Because of this, several of the variables were normalized. (This is the process of transforming the underlying variable values into Z-scores, or the distance of an observation from the mean of the variable.) Below are plots of four of the chemical measurement variables in their untransformed state.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/winesales/chem_distributions.png" alt="chemical variable distributions">

The distributions have very little skew (the majority of the values are found near the mean), but they are still not normal distributions. The presence of a large number of “balanced” outliers on both the left and right side of these distributions could pose issues for modeling. For this reason, each of the chemical measurement variables were normalized and denoted by a leading “NORM_”.

Both the original variables as well as their normalized version were retained and introduced to the model building process, rather than removing the original untransformed variables. My approach was to use various feature selection methods, like decision trees and stepwise selection, to determine which version of each variable should be used in the final models. None of the variables had heavily skewed distributions or the presence of outliers, apart from what was mentioned above regarding the chemical measurement variables. Accordingly, it was not deemed necessary to make any other transformations to the data, except for the modifications to the TARGET variable, which were briefly mentioned in the Data Exploration section.

TARGET was also modified into a new variable named TARGET_AMT, where all the values were subtracted by one and then the zero values converted to missing values. This was done to assist in the zero inflated Poisson and negative binomial generalized linear models’ ability to interpret the target. The value that was subtracted out of TARGET to created TARGET_AMT was added back in to the final predictions, for those models that used the TARGET_AMT variable.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/winesales/Histogram__TARGET.png" alt="target histogram">

The TARGET_AMT transformation resulted in the histogram above, a classic zero inflated distribution, where a large number of observations have a zero value. Distributions of this kind can pose issues for traditional Poisson models. Sometimes you have to first create a survival/hurdle model where the target is modeled first as being zero or non-zero, and then the incremental counted increase above zero is modeled separately. I used this method as well as several other modeling approaches, and then compared the accuracy of each of their results.

<br/><br/>
<img src="{{ site.url }}{{ site.baseurl }}/assets/images/winesales/grapes.jpg" alt="grapes">
## Model building
Several different models and techniques were performed on the previously mentioned variables, including Linear Regression, Logistic Regression, Negative Binomial Generalized Linear Model (GLM), GLM Hurdle/Survival Model, Zero Inflated Poisson (ZIP) GLM, XGBoost Linear Regressor, XGBoost Classifier, and the XGBoost Tree Regressor. One of the main advantages of the Linear Regression and GLM models is their simplified interpretability and relatively greater ease in being communicated to stakeholders. However, gradient boosting models, such as those created by the XGBoost algorithm, are some of the more intuitive machine learning modeling techniques. Because of the grossly different logical and operational approaches of these two groups of models, it makes sense to compare the results of each against their “peers” within their group; namely GLM versus XGBoost.

### Generalized Linear Models (GLM)
The initial feature selection process for the GLM-based modeling methods was highly dependent on the results from Stepwise variable selection. It also seemed more appropriate to compare all of the GLM models on the same “playing field” using the same variables in each model, rather than perform feature selection separately for each model instantiation. Below are the linear independent variable coefficients for each of the GLM models. New out-of-sample data can be input directly into these equations to produce new Target variable predictions. In the case of linear regression, the target prediction for a specific observation can be obtained by adding the y-intercept and the products of each coefficient and its respective variable’s value for that observation. The magnitude and sign of the coefficient helps to add further data interpretation, showing how each variable is related to the target variable. For example, with all of the other variables held constant, the presence of a M_STARS equal to one, in the Linear Regression model, reduces the number of wine cases sold by 2.24 cases on average, while each unit increase in IMP_STARS increases the number of cases sold by 0.78 on average. Both of these conclusions are logical and fall within the assumed effect STARS has on sales.

Most of the other variables’ coefficients are much harder to interpret logically, in how they should or should not affect wine sales. These variables, mostly chemical measurement variables, also are not that consistent across the different GLM models, either in directional effect on the target (positive or negative) or magnitude of the coefficient. The results for the ZIP and Pzero models are inherently different from the rest because they are modeling different target variables than the others.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/winesales/glm_coefficients.jpg" alt="GLM Coefficients">

Because the target variable in this dataset was originally configured as a counting variable, one might assume that a modeling technique specifically tailored for such a variable type would be the most accurate. However, this was not found to be the case. Negative Binomial Regression is specifically tailored for counting variable distributions, but due to the large amount of zero values in this dataset (referred to as zero inflated data), other methods like survival models could perform better. Survival models combine the predictions of two different models together in the attempt to create a more accurate final prediction.

After the probability of an observation being a zero or non-zero was calculated, the non-zero probability prediction was multiplied by the target amount prediction, in order to obtain a probability-weighted target amount, in theory creating a better predictive accuracy. The ZIP and Pzero coefficients above represent those two separate models that are then combined to create the prediction output of the Hurdle model. The ZIP portion of the hurdle model is a special case of Negative Binomial Regression tailored for Zero Inflated Poisson distributions. The coefficients for this model were the most volatile of all the models, and was arguably the least logical and most difficult to interpret.

### Gradient Boosting (XGBoost)
Gradient boosting is the machine learning process, by which several weak prediction models are combined into one strong ensemble model, typically using decision trees, in order to boost regression and classification performance. The approach is based on the evidence that several weak (shallow) decision trees can learn different relationships within the data better than an integrated fully-grown (deep) decision tree. While the method of "ensembling" is used in both gradient boosting and random forests, boosting models return errors composed of high bias and low variance, whereas random forests usually result in low bias and high variance. [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_intro.html) is a gradient boosting algorithm that has been developed into a Python package. Its success at predicting the number of wine cases purchased was very competitive when compared with the more traditional GLM models.

All of the imputed variables and their transformations were fed into several different permutations of the XGBoost model. Models were fit with the objective parameter set to Poisson and Linear Regression, with the Linear performing far better. 4-fold cross validation was performed on all models to measure root-mean-squared-error (RMSE). In other words, the error results were gathered via robust in-sample
and out-of-sample comparison, in order to ensure the models would perform as well on new data as they did on the training data. Cross validation is actually fairly easy to perform with XGBoost, as the Python package is a wrapper of the [scikit-learn](https://scikit-learn.org/stable/) package, which already has several common machine learning functions built in, including cross validation error measurement. 20% testing data holdbacks were used for validation.

Since there are so many different parameters that can be set during the gradient boosting process, grid search and randomized search were deployed in order to explore whether or not the parameter settings were a limiting factor on the models’ performance. It did not prove to have a significant effect, even after exploring 1000s of different parameter combinations, adjusting the learning rate, number of estimators, amount of subsampling per tree, and depth of trees grown. Sometimes these types of parameter settings can be tuned in such a way that causes severe overfitting of the training data, but with the use of cross-validation this effect was minimized.

Interestingly, the final predictor variables selected by XGBoost not only agreed with which variables were previously theoretically assumed to have the most predictive value, but they also fell in the same anticipated rank of importance. This is with the exception of AcidIndex and IMP_TotalSulfurDioxide, which were not assumed to have either a positive or negative effect on wine sales.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/winesales/tree_structure.jpg" alt="tree structure">

Above is the visual depiction of one of the shallow trees that was grown by the best performing XGBoost model using the four predictor variables that were previously mentioned. Each of the trees that were analyzed had similar construction with IMP_STARS creating the first level branch, followed by LabelAppeal, with various mixtures of the specific variable values used during the decision process. The final leaf values are then aggregated together across all of the trees to create the prediction values for each observation fed into the model.

While these trees are easy to logically follow and understand, they aren’t that easy to interpret. Specifically why each tree branch splits the way it does, with the variable values where they are for that split, can be difficult to interpret on it’s own; and when you are forced to interpret the combination of several of these types of trees the difficulty is compounded. There are significant interpretational benefits in tree-based approaches such as this, when compared with neural networks, which anonymize the identity of the variable values during the network solving process. But they are still not as easily interpreted as the more simplistic, traditional GLM models created previously.

<br/><br/>
<img src="{{ site.url }}{{ site.baseurl }}/assets/images/winesales/wine_barrels.jpg" alt="wine barrels">
## Model selection
If performance is set as the most important driver for model selection, then the XGBoost algorithm produced the most robust and accurate predictions on the most consistent basis. The output from the various Generalized Linear Models was far more volatile, with error rates highly dependent on the specific variables included in each model. But if interpretability is at the forefront of concern, Linear Regression or Logistic Regression were the leading options.

Below are the results from each of the models, as measured by their Root Mean Squared Error (RMSE), or the square root of the sum of squares of the differences between predicted and actual Target values. (If that sentence confused you, then *I'm surpised you've made it this far!*... its basically a mathematical technique to measure how different the model’s predictions were from the actual values it was trained to predict, with disregard for whether or not those predictions were too high or too low.) In the event where these predictions are used to forecast a wine producer’s potential demand for a specific case of wine, the error term could be adjusted to more heavily penalize predictions that are too low and result in potential missed sales opportunities. RMSE was also used for comparison because of its easy translatability across various prediction modeling methods. Since we are comparing GLMs and XGBoost, a flexible and common comparison needs to be used to appropriately measure the two groups against each other.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/winesales/model_scores.jpg" alt="model scores">

The variability of RMSE for the GLM models was much greater than that found for the XGB methods. The best of the GLM models was surprisingly the Linear Regression model. One thing that could be attempted to improve the accuracy of the GLM models is a more exhaustive exploration of the various predictor variable combinations. The GLM methods could perform better with a different combination of features/predictors, but with the drastically better RMSE found with the gradient boosting models, this may not be a labor worth investing in.

Interestingly, the XGBoost models performed within a fairly tight band of prediction errors, and the improvements found from an exhaustive grid search of hyper parameters did not yield that much better results than did the untuned tests. This speaks to the inherent value in the algorithm’s approach.


<br/><br/>
<img src="{{ site.url }}{{ site.baseurl }}/assets/images/winesales/red-white-wine.jpg" alt="red white cling">
## Conclusions
While all possible combinations of the variables were not explored for every single Generalized Linear Model technique, the models explored provide enough support that the advanced GLM methods do not provide incremental benefit above simple linear regression. As was initially hypothesized, the chemical predictor variables (as a whole) do not add sufficient predictive power above the STARS ratings and LabelAppeal measurement. It appears evident from this dataset that wine sales are largely dependent on the ratings of wine experts. The XGBoost model provided significant prediction error reduction, and with its fairly easily interpretable tree-based approach, it is our preferred model to predict wine sales using this dataset. But as always, we yield to the applicability to the business, and depending on the business’s sensitivity to error, linear regression could provide sufficient results.

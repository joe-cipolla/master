---
title: "Machine Learning: Wine Sales Prediction"
date: 2019-11-01
tags: [machine learning, regression, xgboost]
header:
  image: /assets/images/winesales/header.jpg
  teaser: /assets/images/winesales/teaser.jpg
excerpt: "prediction model comparison using wine properties"
mathjax: "true"
---
Wine is as subjective a product as any with its various flavors, grapes, producers and production methods. What makes a particular wine taste good to one person may easily be the cause of distaste for another. Here I explore the predictability of wine sales based upon various wine properties, ranging from acidity to label appeal to alcohol content. Fourteen (14) different features are explored and then integrated into several different modeling methods, including regression, hurdle models, and gradient boosting.

## Data Exploration
This dataset includes over 12,000 records of wine sales, each with its own properties as shown in the table below. The theoretical effect of most of the variables is unknown.  Whether or not the chemical composition of a wine (like sulphates/chlorides content) drives a wine's appeal to customers will be explored. However, variables related to experts’ reviews of the wines should have a positive effect on how many cases are sold, as should ratings on the appeal of the wine bottle label increase the number of sales.

The prediction target is a counting variable, ranging from 0 to 8, 0 meaning that no cases of wine were purchased. A variable of this kind necessitates different modeling techniques because of its [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution), as opposed to a binomial or normal. This will require a few transformations to prepare the data for our various modeling techniques.

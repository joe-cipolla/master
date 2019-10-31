---
title: "What's with the name?"
date: 2019-10-30
tags: [data science, statistics, visualizations]
excerpt: "statistical explanation of skewness"
header:
  image: /assets/images/skewed/pisa.jpg
  teaser: /assets/images/skewed/pisa.jpg
---

What does it mean to be skewed to the left? And why would I name my site that?

TL;DR - For those who either still remember their statistics classes, or don't
care to be refreshed: Data distributions with a majority of observations on the
right side of the horizontal axis are referred to as skewed to the left.

In trading and business, PnL distributions with a positive mean and negative skew
(skewed to the left) are the holy grail. Everyone loves to have portfolios
composed mostly of profits with very few losses, even as rare as they tend to be.

<br/><br/>

All the code used to create the following data and visualizations can be found on
my github, under [this](https://github.com/joe-cipolla/skewedtotheleft-explanation) repository.


## The Normal Distribution
Most people have heard of the "normal" Gaussian distribution, often referred to
as the "bell curve" because of it's bell-like shape. A distribution is really
just another way to say "collection" or "group".

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/skewed/normal_distribution.png">

The normal distribution can be described with a few qualities, namely its mean
and standard deviation. It's mean, or average, is 0, and it's standard deviation is 1.
The standard deviation is simply a measure of a dataset's variability, or how
much different you can expect one value in the dataset to be different from another value.
To really understand standard deviation well you need to manually do the math or
follow along with an example like on [wikipedia](https://en.wikipedia.org/wiki/Standard_deviation).

Intuitively, the approach is to calculate the mean (average) of a group of data,
and then measure how different each data point is from that mean. The average difference
is the standard deviation. The real calculation has some number squaring and square-rooting
in order to handle positive and negative differences from the mean.

The plot above was created by taking 10,000 random samples from a group of values
that had a mean of 0 and a standard deviation of 1. It shows the relative likelihood,
or probability, of any one observation value being between -4 and 4. Because the
mean of the distribution is 0, the highest probabilities occur near 0. Because this
distribution is normal, and has a standard deviation of 1, probabilities symmetrically
decrease the farther you get away from 0.

Another way to display a distribution is by using a boxplot. I like bloxplots
because they reduce the amount of information the user has to digest, and therefore
can speed up analysis and decision making. I personally can be distracted very easily
by outliers and distribution tails, spending an inordinate amount of time
hypothesizing and thinking about them, when most decisions are usually made
using the majority of the data in the middle anyways.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/skewed/annotated_box_plot.jpg">

As shown above, a boxplot is composed of:
- a box in the middle, which represents the Interquartile Range (IQR) or 25th to 75th
percentile of the dataset
- a vertical line in the center of the box, representing the median of the data
- a horizontal line, which usually represents the remaining 99% of the data, although
that can vary between 95% to 99% depending on the statistician / programming package.
In other words, 99% of the data can be found between the vertical line on the
far left and the vertical line on the far right.
- dots or symbols on the ends of the horizontal line, signifying the remaining 1% of
data - commonly referred to as outliers

The importance of the normal distribution to statistical theory can't be stressed enough.
Because of the normal distribution and the probabilities implied through it, we
are able to make assumptions about how accurate different numerical models are, and how
different bodies of data will behave. It sits at the core of scientific discoveries
from physics to psychology to computing.

This video does a great job explaining random event sampling and the tendency of
various parts of the natural world to be measured as normal distributions. It's also pretty fun to see
the way they explained this stuff "back in the old days" before modern computing.

Skip to 4:53 to see the bean machine normal distribution demonstration.
<iframe title="vimeo-player" src="https://player.vimeo.com/video/351443264" width="640" height="480" frameborder="0" allowfullscreen></iframe>

<br/><br/>
## The Skewed Distribution
Now that we have our heads wrapped around what "normal" is, we have a better idea of
what makes something not normal. Skewness happens when the peak of the curve of the distribution
starts to lean one way or another, or the top of the "bell". Often the median
and the mean start to diverge as well, and you start to see tails in the distribution
with different sizes.

The plot below shows what negative skewness, or data that is skewed to the left,
looks like.  The majority of the data can be found on the right side of the plot.
The probability of randomly sampling a single value from this distribution is much
higher for higher values. Whereas smaller values are less likely. That, in essence,
is what it means for data to be skewed to the left.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/skewed/left_skewed_distribution.png">

This dataset has a mean of 4.04, a standard deviation of 0.61, and a skew of -0.88. With
a positive mean, and negative skew, values below zero are very unlikely to occur.
Profits and losses (PnL) distributions on trading strategies that look like this
are very appealing. Often times we are happy to find PnL distributions that only
slightly lean to the right. If I were able to establish a trading strategy with a
shape like the plot above (and with literally no downside) I would be a very lucky man.

While many modern applications of machine learning attempt to avoid the statistical assumptions
made by traditional regression methods like [ordinary least squares](https://en.wikipedia.org/wiki/Ordinary_least_squares), there are still plenty of algorithms that require data
be normalized first (transformed to be similar to a normal distribution) before being fed into
the model. Because many of these statistically-reliant approaches are often less
computationally expensive and easier to deploy, it would behoove any data scientist to
at least test their prediction problems using these methods first before running to
TensorFlow or extensive hyper-parameter tuning in XGBoost.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/skewed/left_skewed_boxplot.png">
You may find similarities between this boxplot and my logo, or you may not.

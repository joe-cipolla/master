---
title: "What's with the name?"
date: 2019-10-30
tags: [data science, statistics, visualizations]
excerpt: "statistical explanation of skewness"
header:
  image: /assets/images/skewed/graffiti.jpg
  teaser: /assets/images/skewed/graffiti.jpg
---

What does it mean to be skewed to the left?
And why would I name my site that?

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

The normal distribution can be described with a few qualities, namely its mean
and standard deviation. It's mean, or average, is 0, and it's standard deviation is 1.  
The standard deviation is simply a measure of a dataset's variability, or how
much different you can expect one value in the dataset to be different from another value.
To really understand standard deviation well you need to manually do the math or
follow along with an example like [wikipedia's](https://en.wikipedia.org/wiki/Standard_deviation). Intuitively, the approach is to calculate the mean (average) of a group of data,
and then measure how different each data point is from that mean. The average difference
is the standard deviation. The real calculation has some number squaring and square-rooting
in order to handle positive and negative differences from the mean.

The plot below was created by taking 10,000 random samples from a group of values
that had a mean of 0 and a standard deviation of 1. It shows the relative likelihood,
or probability, of any one observation value being between -4 and 4. Because the
mean of the distribution is 0, the highest probabilities occur near 0. Because this
distribution is normal, and has a standard deviation of 1, probabilities symmetrically
decrease the farther you get away from 0.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/skewed/normal_distribution.png">

The importance of the normal distribution to statistical theory can't be stressed enough.
Because of the normal distribution and the probabilities implied through it, we
are able to make assumptions about how accurate different numerical models are, and how
different bodies of data will behave. It sits at the core of scientific discoveries
from physics to psychology to computing.

This video does a great job explaining random event sampling and the tendency of
the natural world to measured as a normal distribution. It's also pretty fun to see
the way they explained this stuff "back in the old days" before modern computing. Skip to
4:53 to see the bean machine normal distribution demonstration.
<iframe title="vimeo-player" src="https://player.vimeo.com/video/351443264" width="640" height="480" frameborder="0" allowfullscreen></iframe>

<br/><br/>
## The Skewed Distribution
<img src="{{ site.url }}{{ site.baseurl }}/assets/images/skewed/pisa.png">
Now that we have our heads wrapped around what "normal" is, we have a better idea of
what we can do to mess it up. Skewness happens when the curve in the distribution
starts to lean one way or another. Often the median and the mean start to diverge,
and you see different sized tails in the distribution.

The plot below shows what negative skewness, or data that is skewed to the left,
looks like.  The majority of the data can be found on the right side of the plot.
The probability of randomly sampling a single value from this distribution is much
higher for higher values.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/skewed/left_skewed_distribution.png">

This dataset has a mean of 4.04, standard deviation of 0.61, and skew of -0.88. With
a positive mean, and negative skew, values below zero are much less likely to occur.
My concentration on trading over the years has been spent looking for distributions of
profits and losses (PnL) on trading strategies that look like this. Often times we
are happy to find PnL distributions that only slightly lean to the right. 

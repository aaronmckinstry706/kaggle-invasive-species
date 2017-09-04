# An Experiment in Semi-Supervised Learning

This code is inspired by the Kaggle competition [Invasive Species Monitoring](https://www.kaggle.com/c/invasive-species-monitoring). I was attempting to build off of my previous attempt at a competition with a very small training dataset ([The Nature Conservancy Fisheries Monitoring](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring); the code for that is also on GitHub)---however, I quickly realized that I simply wanted to test out an idea of mine (also, I guiltily admit, I didn't have the code ready in time for the competition--so competing wasn't really an option at that point). 

## Pre-Training on Random Labels

My idea was as follows:
1. obtain auxiliary image data from [leafsnap](http://leafsnap.com/dataset/) and [LifeCLEF](http://www.imageclef.org/lifeclef/2016/plant);
2. choose an architecture which performed well on ImageNet (for simplicity, I used the [All-Convolutional Net](https://arxiv.org/abs/1412.6806) here);
3. pre-train that architecture by providing random labels for each image in the auxilliary dataset;
4. fine-tune the weights by training on the auxiliary data. 

I got this idea from [Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/pdf/1611.03530.pdf), where Bengio et al. show that successful architectures--without any form of regularization--can fit a random labelling of the test set. My intuition is that, even with a random labelling of the auxilliary data, the network will still learn good high-level features which can be exploited during fine-tuning. Part of the intuition comes from the idea that, by fitting random labels, the decision boundaries will be concentrated among the densest portions of the data; the distance that these decision boundaries must then move may be much lower than the distance they'd have to move from a random initialization. 

However...I was wrong. Suppose I (1) pretrain the network on a randomly-labeled subset of data, and then (2) train the network normally on the labeled subset of data; if the network could learn useful features by pretraining on randomly-assigned labels, then (once (2) was complete) the network should be achieving an accuracy comparable to a normally-trained instance of the same architecture. The baseline accuracy here is 85-90% for a normally-trained network. Following the steps I outlined, I was barely able to achieve 20% accuracy in the best case. This provides strong evidence that pretraining on randomly-labeled data is pretty terrible for the network. 

In any case, it was worth a try. On to the next idea!

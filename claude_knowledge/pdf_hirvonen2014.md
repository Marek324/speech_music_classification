---
source: "/home/marek/Downloads/Hirvonen2014.pdf"
title: "Speech/Music Classification of Short Audio Segments"
authors: ["Toni Hirvonen"]
aliases: ["hirvonen", "short-segments", "sparse-coding", "short audio segments"]
date_prepared: "2026-05-09"
date_published: "2014"
---

# Speech/Music Classification of Short Audio Segments

**Toni Hirvonen**
Dolby Laboratories, Inc.
Stockholm, Sweden
Email: toni.hirvonen@dolby.com

*Published in: 2014 IEEE International Symposium on Multimedia (ISM), pp. 36–39, 2014. DOI: 10.1109/ISM.2014.18*

---

## Abstract

Research on speech/music classification of digital audio has been both popular in academia, and increasingly utilized in industry. Most of the usual methods use carefully hand-crafted features with Gaussian Mixture Models. To get best performance, some of the features necessitate a long latency due to lookahead, or/and a long onset error. This paper aims to have a different approach to the problem by exploring some of the latest trends in machine learning that have resulted in improvements in other fields. Specifically, it is shown that we can achieve comparable performance by only analyzing segments in the order of tens of milliseconds without the use of following or previous audio. This is done by using a method that allows automatic generation of arbitrarily many features from preprocessed spectrograms.

**Keywords** — audio classification; feature learning; sparse coding;

---

## I. Introduction

Audio, along with video and images, is a domain which in many applications benefits from automatic classification and segmentation to help coping with the immense amount of data. During the last years of increasing computational power, such problems have received even greater interest. A popular higher-level task is to classify audio segments either as speech or music. Being a relatively simple problem with only two classes, and abundance of easily labeled training data, it has been of interest in academia. As an example industrial application, general purpose audio codecs like the recent USAC [1] have utilized such classifiers to switch between speech- and general audio modes with good accuracy [2].

Speech/music classification has so far predominantly been researched from an audio engineering perspective. Typically, an audio expert uses heuristic knowledge-based methods to hand-craft features that intuitively can discriminate between music and speech. The benefit of this approach is good performance while using relatively few features. A widely cited method using 13 features and Gaussian Mixture Models is given in [3]. More recent research has been much focused on refining the complexity. It was shown in [4] that only two features can be sufficient.

However, at least some of the essential traditional features, like zero crossing variance, are calculated over long (in the order of one second) segments of audio. Best results with such features require significant temporal averaging with much lookahead. Even for real-time systems without a significant lookahead, the consequence is a long onset error before sufficient statistics are found. There is also the general issue of having sufficiently long segments of audio available for analysis in the first place. The few studies that have experimented with features requiring only brief audio segments have shown notably diminished accuracy if they are not combined with at least zero-crossings [5, 6].

This paper investigates the task of speech/music classification of short segments in the order of tens of milliseconds in isolation without the use of the previous or the following parts of the signal. Obviously, the long-term traditional features cannot be used. We have thus adopted more of a statistical approach with arguably less heuristics, and find the features in an unsupervised manner from preprocessed spectrograms.

Unsupervised feature learning is a recent trend, and it has shown benefit over older hand-crafted features in fields such as speech recognition [7] and image classification [8]. Especially in speech recognition, the problem is to classify at the phoneme level, which necessitates the analysis of short audio segments. The methods used for the more difficult speech recognition tasks could then be assumed to be suitable for the present task as well.

A common understanding is that the development of deep neural network architectures has been one of the main reasons for the improvements over previous hand-crafting approaches. However, it is not completely understood what are all factors behind these improvements. For example, in [9] it was shown that methods with less demanding feature training procedures can achieve comparable results with state-of-the-art, provided adequate feature encoding and total number of features. Additionally, the accuracy of a machine learning method depends on the implementation details, such as the data preprocessing applied.

Since the submission of the first version of this paper, a recent work has adopted a somewhat similar philosophy to the speech-music classification problem. In [10], the intention was to adapt the commonly utilized deep neural networks for the task. Performance was measured mainly on the level of short frames, while utilizing simple pooling of decisions to analyze longer segments. The raw data utilized is either spectrogram-based like here, or Mel-cepstrum style. The differences to the present work are most notably: 1) different preprocessing, with the lack of compression and higher frequency range used here, and 2) the deep network architecture requiring more expensive training than the framework detailed below. An average error of approx. 8% was given in [10] at the level of 90 ms segment length and three non-overlapping frames.

---

## II. Classification Framework

As the classification framework, we use a similar method as Coates and Ng for image classification [9]. They compared several methods for feature learning, like networks of Restricted Boltzman Machines usual for Deep Learning architectures. Based on their low complexity and state-of-the-art performance, we have investigated Sparse Coding (SC) [11] and a related variant using "one-hot" Orthogonal Matching Pursuit (OMP) [12]. We experimented heuristically with various parameter values and detail those that maximized accuracy.

Given a input signal vector $x \in \mathbb{R}^n$, a feature dictionary matrix containing $d$ features at the columns, $D \in \mathbb{R}^{n \times d}$, both methods find the features by iteratively minimizing over both $D$ and the feature combination $c$ in turn. Additionally, the features are always normalized by their respective L2 norms.

The cost function to be optimized for SC is:

$$\min_{D,c} ||Dc - x||_2^2 + \lambda ||c||_1 \quad (1)$$

over all training samples. The characteristic L1 penalty is adjusted by $\lambda$, which is this study was set to 1. The coordinate descent algorithm [13] was used to solve $c$ in the SC method.

For the OMP method, the cost function is similarly:

$$\min_{D,c} ||Dc - x||_2^2, \quad \text{subject to } ||c||_0 = 1. \quad (2)$$

Thus OMP chooses at each iteration the feature with the largest inner product with the signal, making $c$ extremely sparse. This allows fast generation of large dictionaries.

In [9] it was concluded that with this kind of framework, the exact nature of the features is actually less important than having a suitable method of *encoding* the input data in terms of the features. This is clearly a different principle than with the traditional hand-crafting approach. For discriminability, it is beneficial to introduce sufficiently many features and an encoding mechanism that imposes sparsity. Using Sparse Coding to produce $c$ as in (1) is one natural choice (L1 penalty being more manageable than L0).

Alternatively, a simple soft threshold method with a good performance was discussed in [9]. The encoding is simply obtained by taking the inner product between the signal and the features, $D^T x$, and setting the values whose absolute magnitude is below fixed threshold $\alpha = 0.25$, to zero. This imposes discriminating sparsity, and the method was shown to have comparable performance to SC.

Given the feature dictionary the encoding method, and labeled data, one can train any standard classifier. Since the present method uses a relatively large dimensionality of discriminating features, we have used a linear Support Vector Machine (SVM) with L2 loss function. The SVM penalty factor $C$ was set to 100. If the number of features is increased to several thousand, Coates recommends a smaller value.

---

## III. Classifier Training and Evaluation

### A. Segmentation and Preprocessing

The database used for training and evaluation included a separate music (many styles, 242 unique files, total 9.4 hours) and speech set (many languages and talkers, 282 files, total 6.6 hours). It can always be questioned if such datasets include sufficient variability. On the other hand, the training set can also be limited on purpose for a given application. The present datasets attempt to capture a large range of realistic situations that a typical classifier operating for multimedia content would encounter. We have not included noisy speech (e.g. poor quality microphone capture), or more experimental styles of music.

The raw input data for the feature training and classifier were spectrograms using a 20 ms window size, with a 10 ms frame overlap if longer than 20 ms total segment length was used. As first step, we calculate the compressed log signal power in logarithmically spaced frequency bands from the signal Short-Time Fourier Transform $X$. Comparing logarithmic banding to bin grouping based on Equivalent Rectangular Bandwidth did not seem to affect results notably. The number of bands can be varied according to complexity requirements, here it was 50. With fewer bands, the performance started degrading. For each band of index $i$ containing bins $b_i$, the value of the final spectrogram $S$ was:

$$S(i) = 10 \log_{10}\left( \sum_{b_i} (|X(b_i)|^2) / \text{length}(b_i) + 1 \right). \quad (3)$$

Successive time frame values are concatenated to form a single vector of specrogram values per segment.

Training of the classifier proceeds with the following steps: The gathered band vector data is normalized in terms of mean and standard deviation across spectrogram bands, and whitened (transformation resulting in an identity covariance matrix) using Zero-Phase Component Analysis (ZCA). This is a typical machine-learning step, and here it removes the redundant information between spectrogram bands and equalizes the components. After this, the feature dictionary is learned with the desired feature learning method with this new data as input. Training stage encoding is obtained by applying the dictionary to the data used for feature learning according to the encoding method (SC or threshold). As in [9], we separate the positive and negative values of the encoding vector as separate and concatenate their absolute values to form the final encoding. Encoding data is normalized in terms of mean and standard deviation, and used along with labeling information to train the final SVM classifier.

Test step involves similar gathering of segment spectrograms, which are ZCA whitened and normalized using the training data covariance, mean, and standard deviation. The same feature encoding as previously is applied and the result is normalized. The final classification is obtained from the previously trained SVM.

### B. Results

We detail the results for the combination of OMP feature training and soft threshold encoder. Interestingly, this seemed to produce the best results with a small benefit over SC learning or encoding.

Table 1 shows a cross-validation run where each case had a different training and evaluation set. Any one set had half a million randomly chosen segments (equal number of speech and music), totally silent segments not allowed. The stereo music was summed to mono. We report the classification percentage accuracy with number of features 50–1000 and segment lengths 20–100 ms (1–9 overlapping frames of 20 ms). The accuracy in the training phase was very close to the reported test accuracy. The results seemed to represent a typical run, since little variance in percentages was noticed in repetitions of the procedure.

Increasing the segment length and the number of features both help performance. If feasible, using more than 1000 features is expected to give further improvements. Utilizing temporal variance information by using several 20 ms frames seems preferable. For typical real-time speech/music classifiers with the long lookback, 95% can be considered comparable [3, 4]. Also, the few results for short segments seem to generally be of less accuracy than in the present study [5, 10]. It should be noted that direct comparison is difficult due to different databased used in each study. All in all, the present method seems to provide a good accuracy for short segments of audio.

We also experimented with seeing how the present method copes with over-fit to a more limited set. Using a half of the databases' files as the training set, and the other half of the test set, resulted in a small degradation (maximum approx. 2%) in test accuracy. The training accuracy was on the other hand increased by a similar amount, which indicates over-fit and too limited training. The split was purposefully chosen so that talkers, genres etc. were not overlapping in training and testing. The classification framework can be expected to be somewhat robust, since the over-fit effect was not very severe. However, as mentioned earlier there is no real guarantee that the performance would stay exactly similar as in Table 1 if the music and speech databases were extended with arbitrary or more obscure signals, for example noisy cases.

Finally, even though outside the scope of the main problem, the methods used here can be extended to classifying long segments. Most simply, the classifier can be used for shorter segments as here, and some form of temporal averaging or hysteresis of the segment decisions can be applied to obtain a final classification. As expected based on similar tests in [10], informal experiments of this nature show the error being very low for longer segment lengths. Refining these methods will be part of the related future work.

**Table I**: Cross-validation results for the classification framework over random half a million test segments over different number of features and segment lengths.

| | 20 ms | 40 ms | 60 ms | 80 ms | 100 ms |
|---|---|---|---|---|---|
| 50 features | 87.3 | 89.0 | 89.7 | 90.5 | 91.0 |
| 100 features | 88.7 | 91.5 | 92.4 | 92.7 | 93.4 |
| 500 features | 91.3 | 94.1 | 95.1 | 95.6 | 95.9 |
| 1000 features | 92.1 | 94.8 | 95.8 | 96.2 | 96.5 |

---

## IV. Conclusion

This paper presents a method for speech/music classification using unsupervised feature learning from raw spectrograms, as well as sparse feature encoding. This allows the use of short audio segments, without resorting to lookahead or lookback, while retaining good classification accuracy. Performance can be improved by having more features and/or analyzing a longer temporal segment. The seeming drawback compared to prior art is the increased complexity associated with the larger number of features. Cheap matrix operations can nevertheless make the method feasible for even real-time operation, as learning the features is performed off-line.

---

## References

[1] ISO/IEC 23003-3:2012 *Information technology — MPEG Audio Technologies — Part 3: Unified Speech and Audio Coding*, 2012.

[2] ISO-IEC, MPEG-4 Overview (ISO/IEC JTC1/-SC29/WG11 N2995 Document), http://www.cselt.it/mpeg/standards/mpeg-4/, Oct. 1999.

[3] E. Scheier and M. Slaney, *Construction and Evaluation of a Robust Multifeature Speech/Music Discriminator*, Proc. IEEE Conf. on acoustics, Speech, and Signal Processing (ICASSP), 1997.

[4] C. Panagiotakis and G. Tziritas, *A Speech/Music Discriminator Based on RMS and Zero-Crossings*, IEEE Transactions on Multimedia, Vol. 7, No 1, 2004.

[5] K. El-Maleh, M. Klein, G. Petrucci and P. Kabal, *Speech/music Discrimination for Multimedia Applications*, Proc. IEEE Conf. on acoustics, Speech, and Signal Processing (ICASSP), 2000.

[6] J. Enrique Muoz Expsito, S. Garcia-Galn, N. Ruiz-Reyes, P. Vera-Candeas and F. Rivas-Peal, *Speech/Music Discrimination Using a Single Warped LPC-Based Feature*, Proc. International Society for Music Information Retrieval (ISMIR), 2005.

[7] G. Hinton et al, *Deep Neural Networks for Acoustic Modeling in Speech Recognition*, IEEE Signal Processing Magazine, Vol 29, No 6, 2012.

[8] O. Russakovsky et al, *ImageNet Large Scale Visual Recognition Challenge*, arXiv:1409.0575, 2014.

[9] A. Coates and A. Y. Ng, *The Importance of Encoding Versus Training with Sparse Coding and Vector Quantization*, Proc. 28th International Conference on Machine Learning (ICML), 2011. Code available in http://www.cs.stanford.edu/acoates/.

[10] A. Pikrakis and S. Theodoridis, *Speech-Music Discrimination: a Deep Learning Perspective*, Proc. 22nd Europen Signal Processing Conference (EUSIPCO), 2014.

[11] B. A. Olshausen and D. J. Field, *Sparse Coding with an Overcomplete Basis Set: A Strategy Employed by V1?*, Vision Res., Vol 37, No 23, 1997.

[12] Y. C. Pati, R. Rezaifar and P S. Krishnaprasad *Orthogonal Matching Pursuit: Recursive Function Approximation with Applications to Wavelet Decomposition*, Proc. 27th Annual Asilomar Conference on Signals, Systems, and Computers, 1993.

[13] T. T. Wu and K. Lange, *Coordinate Descent Algorithms for Lasso Penalized Regression*, Annals of Applied Statistics, Vol 2, No 1, 2008.

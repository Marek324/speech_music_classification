---
source: "/home/marek/Downloads/Bhattacharjee2018.pdf"
title: "Time-Frequency Audio Features for Speech-Music Classification"
authors: ["Mrinmoy Bhattacharjee", "S.R.M. Prasanna", "Prithwijit Guha"]
aliases: ["bhattacharjee", "sps", "time-frequency features", "spectral peak sequences"]
date_prepared: "2026-05-09"
date_published: "2018-11-03"
---

# Time-Frequency Audio Features for Speech-Music Classification

**Mrinmoy Bhattacharjee**, *Student MIEEE* — Dept. of Electronics and Electrical Engineering, Indian Institute of Technology Guwahati, Guwahati-781039, India
**S.R.M. Prasanna**, *SMIEEE* — also with Dept. of Electrical Engineering, Indian Institute of Technology Dharwad, Dharwad-580011, India
**Prithwijit Guha**, *MIEEE* — Dept. of Electronics and Electrical Engineering, Indian Institute of Technology Guwahati

email: {mrinmoy.bhattacharjee, prasanna, pguha}@iitg.ac.in

*arXiv:1811.01222v1 [eess.AS] 3 Nov 2018*

---

## Abstract

Distinct striation patterns are observed in the spectrograms of speech and music. This motivated us to propose three novel time-frequency features for speech-music classification. These features are extracted in two stages. First, a preset number of prominent spectral peak locations are identified from the spectra of each frame. These important peak locations obtained from each frame are used to form Spectral peak sequences (SPS) for an audio interval. In second stage, these SPS are treated as time series data of frequency locations. The proposed features are extracted as periodicity, average frequency and statistical attributes of these spectral peak sequences. Speech-music categorization is performed by learning binary classifiers on these features. We have experimented with Gaussian mixture models, support vector machine and random forest classifiers. Our proposal is validated on four datasets and benchmarked against three baseline approaches. Experimental results establish the validity of our proposal.

**Index Terms** — Time-frequency audio features, speech music classification, spectrogram, SVM

---

## I. Introduction

Content based audio indexing and retrieval applications often involve an important preprocessing step of segmenting and classifying audio signals into distinct categories. Apart from general environmental sounds, speech and music are two important audio categories. Preprocessing steps necessarily require classification algorithms that ensure homogeneity of the category in audio segments. This work focuses on proposing features for better discrimination of speech and music for such audio segmentation applications.

Researchers have observed several differences in speech and music signals. For example, pitch in speech usually exists over a span of 3 octaves only, whereas music consists of fundamental tones spanning up to 6 octaves [1]. Also, specific frequency tones play an important part in the production of music. Hence, unlike speech, music is expected to have strict structures in the frequency domain [2]. Furthermore, short silences usually punctuate speech sound units [3], while music is generally continuous and without breaks (Figure 1). Literature in the *classification of speech and music* (CSM, henceforth) includes many studies that exploit such (and other) differences between them [4, 5]. We briefly review a few closely related works next.

Table I lists the most widely used feature sets of CSM literature. We have categorized these features into two groups viz. *Spectral Features* and *Temporal Features*. Most widely used features from the spectral group are Zero-Crossing Rate (ZCR, henceforth) [2], Spectral Centroid, Spectral Roll-off and Spectral Flux [6]. Energy [7], Entropy [8] and Root Mean Square (RMS) [2] values are the most popular ones from the temporal group. Apart from these, few works have used spectrograms as features and processed them as images. For example, the approach proposed by Mesgarani et al. [9] is inspired by auditory cortical processing and uses Gabor-like spectro-temporal response fields for feature extraction from spectrogram. On the other hand, Neammalai et al. [7] performed thresholding and smoothing on standard spectrograms to form binary images and used them as features for classification. Existing works on speech-music classification have mostly employed Gaussian Mixture Models (GMM) [2, 10, 11], Artificial Neural Networks (ANN) [8], k-Nearest Neighbors (kNN) [12, 13, 14] and Support Vector Machines (SVM) [11, 6, 7] as classifiers. Recent works have also used deep learning techniques for this task [15, 16]. Most existing works have attempted to characterize speech or music using pure temporal and/or spectral features. We believe that time-frequency feature based representations are necessary for better speech-music classification. Our motivation for this proposal is described next.

Figure 1 shows the spectrograms of speech and music. In case of speech, pitch and harmonics slowly change from one frame to another [17]. This leads to the formation of smooth arc-like patterns in its spectrogram. On the other hand, pitch and harmonics in music remain stationary for some finite duration before performing sharp transitions [18]. As such, music spectrograms contain patterns in the form of many horizontal line segments. These can be attributed to the following reasons.

**Inertia of speech production system** — Speech production system possesses inertia [19, 20]. It requires a finite amount of time to change from one sound unit to another, leading to the formation of slowly changing striation patterns in speech spectrogram. Whereas, individual notes of music have a specific onset instant, marked by a relatively large burst of energy that make its striation patterns discontinuous [21].

**Slowly decaying harmonics in music** — Music tones decay slowly. Comparatively, speech production system is a damped system where sound decays quite fast [22, 23].

**Range of sounds produced** — A musical instrument produces only a fixed number of tones and their overtones. On the other hand, speech production system generates a large number of intermediate frequencies while transitioning from one sound unit to another [24, 2].

The tempo-spectral properties of speech and music are quite distinct. Hence, features capturing joint variations in temporal and spectral domains should be harnessed for efficient classification of speech and music. Existing works in this area has used combinations of temporal and spectral audio features [2, 6, 8, 10, 25] for achieving better performance.

We propose three new audio features capable of capturing the joint tempo-spectral characteristics of an audio segment. Peaks in the spectra of audio frames appear as striation patterns in spectrograms. Prominent spectral peaks having relatively higher amplitudes correspond to the brightest patterns in spectrograms. We believe that the frequency locations of such prominent peaks carry class specific information. Accordingly, We compute the features in a two-stage approach. First, these prominent spectral peaks are identified in all frames of an audio interval. Second, locations of detected peaks across frames are treated as temporal sequences, defined as spectral peak sequences (SPS). The proposed features are derived as zero crossing rate, periodicity and second order statistics of each SPS. The speech-music classification is performed by training classifiers on these features. The proposed scheme for feature extraction is described in further detail in Section II.

We have benchmarked our proposal on four audio datasets and against three baseline approaches [10, 2, 6]. The results of our experiments are reported in Section III. Finally, we conclude in Section IV and sketch the possible future extensions of the present proposal.

**Table I**: Most widely used audio features in speech vs music classification literature

| Group | Features | Papers |
|---|---|---|
| Spectral Features | ZCR, Spectral Centroid, Spectral Flux, Spectral Rolloff, MFCC, Chroma, Log Mel spectrum energy, Harmonic ratio, Modulation spectrum energy, Pitch | [10], [6], [26], [8], [2], [27], [5], [28] |
| Temporal Features | Energy, Entropy, RMS, Peak-to-Sidelobe ratio (PSR) from the Hilbert Envelope of the LP Residual, Normalized Autocorrelation Peak Strength (NAPS) of Zero frequency filtered signal | [10], [6], [8], [7], [2], [25], [5], [29], [28] |

*[Figure 1: Spectrograms of (a) Speech and (b) Music. Note the distinct striation patterns of speech and music. This observation motivated our proposal of time-frequency audio features for speech-music discrimination.]*

---

## II. Proposed work

The audio segment $\mathbf{x}$ ($\mathbf{x}[n] \in \mathcal{R}; n = 0, \ldots, N_s - 1$) is divided into $L$ overlapping frames $\mathbf{x}_l$ ($l = 0, \ldots, L - 1$) of size $2N_f$. Let,

$$\mathbf{X}_l[k] = \sum_{m=0}^{2N_f - 1} \mathbf{x}_l[m] e^{-j k \frac{2\pi}{2N_f} m} \quad (k = 0 \ldots 2N_f - 1)$$

be the DFT of $\mathbf{x}_l$. These frames ($\mathbf{x}_l$) are sequences of real numbers. Hence, we consider only the first half of DFT coefficients (i.e. $\mathbf{X}_l[k]$; $k = 0, \ldots N_f - 1$) from each frame.

The proposed features are extracted in two stages and are described next.

The first stage identifies the important spectral peaks present in each frame of the audio interval. The frequency locations of all spectral peaks in the $l^{th}$ frame are stored in a set $\mathbf{H}_l$. This set is constructed as

$$\mathbf{H}_l = \{k : [\mathbf{X}_l[k - 1] < \mathbf{X}_l[k]] \wedge [\mathbf{X}_l[k] > \mathbf{X}_l[k + 1]]\} \quad (1)$$

where $0 \leq k < (N_f - 1)$. The number of spectral peaks ($|\mathbf{H}_l|$) varies in each frame varies. Thus, we retain at most $p$ prominent spectral peaks from each frame to construct the truncated set

$$\mathbf{tH}_l = \left\{ k_0^{(l)}, k_1^{(l)}, \ldots k_{p-1}^{(l)} : \mathbf{X}_l[k_0] \geq \mathbf{X}_l[k_1] \geq \ldots \geq \mathbf{X}_l[k_p] \right\}$$

However, if $|\mathbf{H}_l| = q < p$ then, the last frequency location $k_{q-1}$ is repeated $p - q$ times to maintain uniformity in cardinality of $\mathbf{tH}_l$ for all frames. The elements of $\mathbf{tH}_l$ are further sorted in descending order to construct the vector

$$\mathbf{pH}_l = \left[ k_{(0)}^{(l)}, k_{(1)}^{(l)}, \ldots k_{(p-1)}^{(l)} \right] \quad (k_{(0)}^{(l)} \geq k_{(1)}^{(l)} \geq \ldots \geq k_{(p-1)}^{(l)})$$

These vectors ($\mathbf{pH}$) are used to construct a $p \times L$ peak sequence matrix $\mathbf{S}_{peak} = \left[ \mathbf{pH}_0^T, \ldots \mathbf{pH}_{L-1}^T \right]$ for an audio interval. Each row of $\mathbf{S}_{peak}$ is defined as a *Spectral Peak Sequence* (SPS, henceforth). It is noteworthy that, the first row of $\mathbf{S}_{peak}$ corresponds to the SPS with highest frequency locations and the last row corresponds to one with lowest frequency locations.

In second stage, the proposed features are extracted from $\mathbf{S}_{peak}$. For notational convenience, the index $r$ ($0 \leq r < p$) will be used for referring to the $r^{th}$ row of $\mathbf{S}_{peak}$ or the $r^{th}$ SPS. Attributes derived from the $r^{th}$ SPS will also be indexed by $r$. This work proposes three different features derived from the SPS. These are **(a) SPS Periodicity (SPS-P, henceforth)**, **(b) SPS Zero Crossing Rate (SPS-ZCR, henceforth)**, and **(c) SPS Standard Deviation, Centroid and its Gradient (SPS-SCG, henceforth)**. The following are computed from the SPS for feature extraction. Let $\mu_r = \frac{1}{L} \sum_{l=0}^{L-1} \mathbf{S}_{peak}[r][l]$ be the centroid frequency location of the $r^{th}$ SPS. These centroid frequencies are used to construct the zero-centered SPS $C_r$ such that $C_r[l] = \mathbf{S}_{peak}[r][l] - \mu_r$ ($l = 0, \ldots L - 1$). The auto-correlation sequence of $C_r$ can be estimated as

$$A_r[\tau] = \frac{1}{L} \sum_{l=0}^{L-1-\tau} C_r[l] C_r[l + \tau]$$

where, $\tau = 0, \ldots \mathcal{L}$ ($\mathcal{L} = \frac{L}{2}$ if $L$ is even and $\frac{L+1}{2}$ otherwise). One or more of these attributes are used to compute the proposed features.

**SPS-Periodicity** — It is well known that quasi-periodic voiced sounds constitute a major part of the speech signals [30, 31]. Whereas, music is created by musicians using their personalized styles of arranging sound items from multiple instruments. Hence, music signals need not necessarily have a periodic nature. Figures 2(a)–(e) show the average trends in autocorrelation sequences of different speech and music SPS estimated from the GTZAN dataset. Presence of peaks (other than the first one) in autocorrelation sequence of a signal indicates its periodicity. Such peaks are observed in autocorrelation sequences of SPS of speech but, not in that of music. This motivated us to exploit the periodicity of SPS as feature for speech-music discrimination. Periodicity of the $r^{th}$ SPS is estimated using its auto-correlation sequence. The peak locations $\tau^{(r)}$ of $A_r$ are detected (Equation 1) and stored in a set $\mathbf{T}_r = \left\{ \tau_0^{(r)}, \tau_1^{(r)}, \ldots \right\}$ ($|\mathbf{T}_r| < \mathcal{L}$) in an ascending order. We compute the quantities $\Delta \tau_u^{(r)} = \tau_u^{(r)} - \tau_{u-1}^{(r)}$ ($u = 1, \ldots |\mathbf{T}_r| - 1$). The variance $V_r$ of these quantities $\{\Delta \tau_u^{(r)}\}$ provides an estimate of the periodicity of the $r^{th}$ SPS. The feature SPS-P is constructed as a $p$ dimensional vector such that $SPS\text{-}P = [V_0, \ldots V_{p-1}]$.

**SPS-Zero Crossing Rate** — Audio signals are non-stationary. Thus, spectral peaks in a certain SPS may correspond to different frequency locations within the spectra of audio frames in an interval. Hence, without any loss of generality, we can assume that spectral peak sequences contain varying values. The Zero Crossing Rate (ZCR) provides a gross estimate of average frequency of time-series data [32]. We propose to compute the ZCR of each SPS to estimate their average frequency and use this as a feature for CSM. The ZCR ($Z_r$) of the $r^{th}$ zero-centered SPS is computed as $Z_r = \frac{1}{2L} \sum_{l=0}^{L-1} |sgn(C_r[l]) - sgn(C_r[l - 1])|$ where, $sgn(\bullet)$ is the signum function. SPS-ZCR feature is constructed as a $p$ dimensional vector such that $SPS\text{-}ZCR = [Z_0, \ldots Z_{p-1}]$. Figures 2(f)–(j) show the distributions of ZCR values for different SPS of speech and music. We observe that, ZCR of lower-frequency SPS (e.g. $Z_{19}$, Figure 2(f)) exhibit significant overlap between the ZCR distribution of the two classes. However, this overlap reduces as music SPS-ZCR values gradually decrease (compared to that of speech) for higher-frequency spectral peak sequences ($Z_{15}$ to $Z_3$, Figures 2(g)–(j)). In general, speech SPS-ZCR values are higher than that of music, indicating that speech SPS values vary more than that of music. Hence, this property can be exploited as a discriminator between the two classes.

**SPS-Standard Deviation, Centroid and its Gradient** — We believe that the frequency locations in any $r^{th}$ SPS are category specific (i.e. either speech or music). This motivated us to propose a set of features based on the statistical properties of the spectral peak sequences. These statistical attributes include the centroid $\mu_r$ and standard deviation $\sigma_r = \sqrt{\frac{1}{L} \sum_{l=0}^{L-1} (\mathbf{S}_{peak}[r][l] - \mu_r)^2}$ of the $r^{th}$ SPS. Also, the rates of change of $\mu_r$ (with respect to $r$) exhibit distinct trends for both speech and music. We compute the gradient $\Delta \mu_r = \frac{1}{2}(\mu_{r+1} - \mu_{r-1})$ for representing this trend. Thus, we propose the SPS-SCG feature as a $3p$ dimensional vector given by $SPS\text{-}SCG = [\mu_0, \ldots \mu_{p-1}, \sigma_0, \ldots \sigma_{p-1}, \Delta \mu_0, \ldots \Delta \mu_{p-1}]$. Here, $\Delta \mu_0 = (\mu_1 - \mu_0)$ and $\Delta \mu_{p-1} = (\mu_{p-1} - \mu_{p-2})$. Figure 2(k)–(m) show the trends of SPS-SCG features averaged over several audio intervals for both speech and music (GTZAN dataset).

The proposed features capture prominent spectral information in the first stage and temporal variations are characterized in the second stage. Binary classifiers are learned on these proposed features. In this proposal, we have experimented with Gaussian mixture models (GMM), support vector machines (SVM) and random forest (RF) classifiers. The results of our experiments with these tempo-spectral features are presented next.

*[Figure 2: Proposed features computed from the GTZAN dataset. (a)–(e) show the trend of autocorrelation sequence $A_r$. Speech $A_r$ indicate presence of periodicity; (f)–(j) show the SPS-ZCR distribution. Speech in general have higher SPS-ZCR values than music; (k)–(m) show the values of $\mu_r$, $\sigma_r$ and $\Delta \mu_r$. Speech and music show distinct trends; Figures represent averaged behavior over the GTZAN data-set. SPS-ZCR and $A_r$ are shown only for $3^{rd}$, $7^{th}$, $11^{th}$, $15^{th}$ and $19^{th}$ SPS of speech and music.]*

---

## III. Experiments and Results

The proposed approach is validated on four datasets. These are **(a)** GTZAN Music/Speech collection [33], **(b)** Scheirer-Slaney Music-Speech Corpus [34], **(c)** Movie dataset, **(d)** TV News Broadcast dataset. The later two datasets are created by us and are available on request for non-commercial usage. The movie dataset consists of 5s clips of pure speech and pure music from old Bollywood movies. The TV News Broadcast dataset contains 5s clips of speech and non-vocal music recorded from Indian English news channels.

Our proposal is benchmarked against the following three baseline approaches. First, the method proposed by Khonglah et al. in [10] (Khonglah-FS). The authors propose that speech specific features like Normalized Autocorrelation Peak Strength of the Zero Frequency Filtered Signal, the Peak-to-Sidelobe Ratio from Hilbert Envelope of the LP residual, Log-Mel Spectrum Energy, and 4-Hz Modulation Energy etc. are better in characterizing speech and hence, good discriminators from music. The second approach proposed by Sell et al. [2] (Sell-FS) uses novel chroma based features that represent music tonality for better speech-music classification. Third, the 13 MFCC coefficients [6] (MFCC) are considered as features as these are widely used in most speech processing applications.

For all our experiments, we have chosen audio intervals of $1 s$ duration. From each audio interval, we have drawn frames of $30 ms$ duration with a shift of $1 ms$. Features are extracted from each audio interval. Accordingly, each audio interval is classified as either speech or music. The number of prominent peaks $p$ is empirically selected and is set to $p = 20$ for all our experiments. We have used MATLAB toolboxes for realizing the GMM and RF based classifiers. The lib-SVM toolbox [35] is used for SVM with radial basis function kernel based classifier. The classifier parameters are optimized by grid-search. The training and test data are chosen in a ratio of 70 : 30. The experiments are repeated 20 times. The mean and variances of F-scores of these independent trials are reported.

The performance of baseline approaches and individual features from our proposal (on GTZAN only) are presented in Table II. SPS-P and SPS-ZCR fail to outperform the baseline approaches. However, SPS-SCG provides a significant improvement over the best baseline. Additionally, we have experimented with early and late feature fusion schemes for our proposal. However, no significant improvement was observed over the performance of SPS-SCG. The comparative performance analysis of proposed features and baseline approaches (with SVM only) for all four datasets are shown in Figure 3. The SPS-SCG features with SVM classifier provides the best performance for GTZAN, Scheirer Slaney and TV News Broadcast dataset. However, it has second best performance for the Movie data-set. Thus, the experimental results establish that the proposed features can effectively capture the time-frequency characteristics of speech and music while discriminating one from another.

**Table II**: Performance of baseline approaches and individual features on GTZAN dataset. Additionally, performances of early and late fusion of proposed features are also presented. Experiments are performed with GMM, SVM and Random Forest. The classifier parameters are optimized by grid-search. SPS-SCG with SVM has better performance compared to baseline approaches and other features.

| | GMM | Random Forest | SVM |
|---|---|---|---|
| Khonglah-FS | 0.91 (0.02) | 0.93 (0.02) | 0.93 (0.01) |
| Sell-FS | 0.94 (0.01) | 0.95 (0.01) | 0.95 (0.01) |
| MFCC | **0.95 (0.01)** | 0.92 (0.01) | 0.97 (0.01) |
| SPS-P | 0.83 (0.05) | 0.86 (0.04) | 0.84 (0.05) |
| SPS-ZCR | 0.81 (0.01) | 0.84 (0.01) | 0.87 (0.03) |
| SPS-SCG | 0.93 (0.01) | **0.95 (0.01)** | **0.98 (0.00)** |
| SPS-EF | 0.93 (0.01) | 0.95 (0.01) | 0.98 (0.00) |
| SPS-LF | 0.91 (0.02) | 0.95 (0.01) | 0.92 (0.02) |

*[Figure 3: The performance of baseline and proposed features on four data-sets using SVM (with radial basis function kernel) classifier. Among proposed features, SPS-SCG has best performance on three out of four datasets. Bars compare F-scores across MFCC, Khonglah-FS, Sell-FS, SPS-P, SPS-ZCR, SPS-SCG on Broadcast News, GTZAN, Movie Dataset and Scheirer Slaney datasets.]*

---

## IV. Conclusion

This work proposes a novel two-stage feature extraction scheme for representing the time-frequency characteristics of an audio interval. In the first stage, we detect the frequency locations of $p$ prominent spectral peaks for each frame in an audio interval. These peak locations are stored as columns in a matrix $\mathbf{S}_{peak}$. The rows of this matrix are defined as the $p$ spectral peak sequences (SPS) that characterize the audio interval. The proposed features are computed in the second stage by treating each SPS as temporal sequence. We estimate the periodicity (SPS-P), ZCR (SPS-ZCR), standard deviation, centroid and its gradient (collectively, SPS-SCG) as features of each SPS. The performance of our proposal is benchmarked on four datasets and against three baseline approaches. The proposed features are deployed with GMM, SVM and Random Forest based classifiers. Among the proposed features, SPS-SCG (with SVM) has better performance compared to baseline approaches and other features on three datasets.

The spectral peak sequences are prominent peak locations (integer values) of frame spectra. This feature can be extended to incorporate sequences of other attributes of frame spectra. The present work focuses on ZCR, periodicity and a few statistical attributes of the spectral peak sequences. This can be further enhanced by considering other temporal sequence features. The proposed features are applied to the domain of speech-music classification. This work can be extended to deploy an enhanced set of these features for effective discrimination of speech, music and multiple categories of environmental sounds.

---

## References

[1] J. Saunders, "Real-time discrimination of broadcast speech/music," in *1996 IEEE International Conference on Acoustics, Speech, and Signal Processing Conference Proceedings*, vol. 2, May 1996, pp. 993–996 vol. 2.

[2] G. Sell and P. Clark, "Music tonality features for speech/music discrimination," in *2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, May 2014, pp. 2489–2493.

[3] C. Panagiotakis and G. Tziritas, "A speech/music discriminator based on rms and zero-crossings," *IEEE Transactions on Multimedia*, vol. 7, no. 1, pp. 155–166, Feb 2005.

[4] V. A. Masouméh and M. B. Mohammad, "A review on speech-music discrimination methods," *International Journal of Computer Science and Network Solutions*, vol. 2, Feb 2014.

[5] Y. Lavner and D. Ruinskiy, "A decision-tree-based algorithm for speech/music classification and segmentation," *EURASIP Journal on Audio, Speech, and Music Processing*, vol. 2009, no. 1, p. 239892, Jun 2009.

[6] E. Mezghani, M. Charfeddine, C. B. Amar, and H. Nicolas, "Multifeature speech/music discrimination based on mid-term level statistics and supervised classifiers," in *2016 IEEE/ACS 13th International Conference of Computer Systems and Applications (AICCSA)*, Nov 2016, pp. 1–8.

[7] P. Neammalai, S. Phimoltares, and C. Lursinsap, "Speech and music classification using hybrid form of spectrogram and fourier transformation," in *Signal and Information Processing Association Annual Summit and Conference (APSIPA), 2014 Asia-Pacific*, Dec 2014, pp. 1–6.

[8] M. Srinivas, D. Roy, and C. K. Mohan, "Learning sparse dictionaries for music and speech classification," in *2014 19th International Conference on Digital Signal Processing*, Aug 2014, pp. 673–675.

[9] N. Mesgarani, M. Slaney, and S. A. Shamma, "Discrimination of speech from nonspeech based on multiscale spectro-temporal modulations," *IEEE Transactions on Audio, Speech, and Language Processing*, vol. 14, no. 3, pp. 920–930, May 2006.

[10] B. K. Khonglah and S. R. Mahadeva Prasanna, "Speech / music classification using speech-specific features," *Digital Signal Processing*, vol. 48, no. C, pp. 71–83, jan 2016.

[11] H. Zhang, X.-K. Yang, W. Q. Zhang, W.-L. Zhang, and J. Liu, "Application of i-vector in speech and music classification," in *2016 IEEE International Symposium on Signal Processing and Information Technology (ISSPIT)*, Dec 2016, pp. 1–5.

[12] J. G. A. Barbedo and A. Lopes, "A robust and computationally efficient speech/music discriminator," *J. Audio Eng. Soc*, vol. 54, no. 7/8, pp. 571–588, 2006.

[13] E. Alexandre-Cortizo, M. Rosa-Zurera, and F. Lopez-Ferreras, "Application of fisher linear discriminant analysis to speech/music classification," in *EUROCON 2005 - The International Conference on "Computer as a Tool"*, vol. 2, Nov 2005, pp. 1666–1669.

[14] J. J. Burred and A. Lerch, "Hierarchical automatic audio classification," *J. Audio Eng. Soc*, vol. 52, no. 7/8, pp. 724–739, 2004.

[15] A. Kruspe, D. Zapf, and H. Lukashevich, "Automatic speech/music discrimination for broadcast signals," in *INFORMATIK 2017*, M. Eibl and M. Gaedke, Eds. Gesellschaft für Informatik, Bonn, 2017, pp. 151–162.

[16] A. Pikrakis and S. Theodoridis, "Speech-music discrimination: A deep learning perspective," in *2014 22nd European Signal Processing Conference (EUSIPCO)*, Sept 2014, pp. 616–620.

[17] Y. Xu and X. Sun, "Maximum speed of pitch change and how it may relate to speech," *The Journal of the Acoustical Society of America*, vol. 111, no. 3, pp. 1399–1413, 2002.

[18] J. F. Alm and J. S. W. Review, "Time-frequency analysis of musical instruments," *Society for Industrial and Applied Mathematics*, vol. 44, no. 3, pp. 457–476, August 2002.

[19] K. S. R. Murty and B. Yegnanarayana, "Epoch extraction from speech signals," *IEEE Transactions on Audio, Speech, and Language Processing*, vol. 16, no. 8, pp. 1602–1613, Nov 2008.

[20] Z. Zhang, "Mechanics of human voice production and control," *The Journal of the Acoustical Society of America*, vol. 140(4), p. 2614–2635, 2016.

[21] J. P. Bello, L. Daudet, S. Abdallah, C. Duxbury, M. Davies, and M. B. Sandler, "A tutorial on onset detection in music signals," *IEEE Transactions on Speech and Audio Processing*, vol. 13, no. 5, pp. 1035–1047, Sept 2005.

[22] J. Meyer, *Structure of Musical Sound*. New York, NY: Springer New York, 2009, pp. 23–44.

[23] L. Oller, S. Ternström, R. I. of Technology. School of Computer Science, M. Communication. Department of Speech, and Hearing, *Analysis of Voice Signals for the Harmonics-to-noise Crossover Frequency*, 2008.

[24] B. K. Khonglah and S. R. M. Prasanna, "Low frequency region of vocal tract information for speech / music classification," in *2016 IEEE Region 10 Conference (TENCON)*, Nov 2016, pp. 2593–2597.

[25] C. Lim and J. h. Chang, "Enhancing support vector machine-based speech/music classification using conditional maximum a posteriori criterion," *IET Signal Processing*, vol. 6, no. 4, pp. 335–340, June 2012.

[26] B. K. Khonglah and S. R. M. Prasanna, "Speech / music classification using vocal tract constriction aspect of speech," in *2015 Annual IEEE India Conference (INDICON)*, Dec 2015, pp. 1–6.

[27] A. Gallardo-Antolin and J. M. Montero, "Histogram equalization-based features for speech, music, and song discrimination," *IEEE Signal Processing Letters*, vol. 17, no. 7, pp. 659–662, July 2010.

[28] A. Pikrakis, T. Giannakopoulos, and S. Theodoridis, "A speech/music discriminator of radio recordings based on dynamic programming and bayesian networks," *IEEE Transactions on Multimedia*, vol. 10, no. 5, pp. 846–857, Aug 2008.

[29] J. H. Song, K. H. Lee, J. H. Chang, J. K. Kim, and N. S. Kim, "Analysis and improvement of speech/music classification for 3gpp2 smv based on gmm," *IEEE Signal Processing Letters*, vol. 15, pp. 103–106, 2008.

[30] A. Biswas, P. K. Sahu, A. Bhowmick, and M. Chandra, "Feature extraction technique using erb like wavelet sub-band periodic and aperiodic decomposition for timit phoneme recognition," *International Journal of Speech Technology*, vol. 17, no. 4, pp. 389–399, Dec 2014.

[31] H. Kawahara, M. Morise, R. Nisimura, and T. Irino, "Higher order waveform symmetry measure and its application to periodicity detectors for speech and singing with fine temporal resolution," in *2013 IEEE International Conference on Acoustics, Speech and Signal Processing*, May 2013, pp. 6797–6801.

[32] D. S. Shete and P. S. B. Patil, "Zero crossing rate and energy of the speech signal of devanagari script," vol. 4, Jan 2014, pp. 01–05.

[33] G. Tzanetakis and P. Cook, "Musical genre classification of audio signals," *IEEE Transactions on Speech and Audio Processing*, vol. 10, no. 5, pp. 293–302, Jul 2002.

[34] E. Scheirer and M. Slaney, "Construction and evaluation of a robust multifeature speech/music discriminator," in *1997 IEEE International Conference on Acoustics, Speech, and Signal Processing*, vol. 2, Apr 1997, pp. 1331–1334 vol.2.

[35] C.-C. Chang and C.-J. Lin, "Libsvm: A library for support vector machines," *ACM Trans. Intell. Syst. Technol.*, vol. 2, no. 3, pp. 27:1–27:27, May 2011.

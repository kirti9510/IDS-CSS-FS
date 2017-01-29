# Feature Selection using Modified Charged System Search for Intrusion Detection Systems in Cloud Environment

DOI: NA

A novel feature selection technique based on a metaheuristic search algortihm is implemented for Intrusion Detection System. The proposed Modified Charged System Search algorithm selects optimal feature subset to give an efficient IDS with higher classification accuracy. The results are evaluated on dataset and presented in the paper. 

## Introduction

The FCM algorithm is very sensitive to noise and is easily struck at local optima. In order to overcome
this limitation, spatial context of connection is taken into account considering its neighboring
connections. A Penalty Reward based FCM algorithm is implemented here which can handle small
as well as large amount of noise by adjusting a penalty and reward coefficient. The algorithm takes
into account both the feature information and spatial information. The objective function is
modified to incorporate the penalty and reward term by which it can overcome the local optima. The new objective function of the PRFCM algorithm is defined as follows:

![alt tag](https://github.com/shakti365/IDS-CSS-FS/blob/master/resources/figures/fig3.png)

### Dataset

The experiments are performed on NSL-KDD and 10% KDD Cup'99 Dataset. These dataset were pre-processed and normalized before use. It can be obtained from the following source. 

[NSL-KDD Dataset](http://www.unb.ca/research/iscx/dataset/iscx-NSL-KDD-dataset.html)
[KDD Cup'99 Dataset](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)

### Results

The following results show the performance MCSS algortihm.

This figure shows variying classification accuracy for different number of features selected during an instance in search thus a need for feature selection.

![alt tag](https://github.com/shakti365/IDS-CSS-FS/blob/master/resources/figures/fig1.jpg) 

This figure shows fast convergence of the MCSS algorithm

![alt tag](https://github.com/shakti365/IDS-CSS-FS/blob/master/resources/figures/fig2.png)

This figure shows postions of different agent during instances of search and its convergence towards the end. The few particles which do not converge are present as an improvement to give more exploration to the search.

![alt tag](https://github.com/shakti365/IDS-CSS-FS/blob/master/resources/figures/barchart.png)

## Authors

* **Shivam Shakti**
* **Partha Ghosh**
* **Santanu Phadikar**

## Copyright

This paper has been accepted and presented in SCESM 2017. 
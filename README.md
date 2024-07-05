# Multi-threshold Deep Metric Learning for Facial Expression Recognition

This repo is the official implementation for **Multi-threshold Deep Metric Learning for Facial
Expression Recognition** The paper has been accepted to Pattern Recognition



## Introduction

Effective expression feature representations generated by a triplet-based deep metric learning are highly advantageous for facial expression recognition (FER). The performance of deep metric learning algorithms based on triplets is contingent upon identifying the best threshold for triplet loss. Threshold validation, however, is tough and challenging, as the ideal threshold changes among datasets and even across classes within the same dataset. In this paper, we present the multi-threshold deep metric learning technique, which not only avoids the difficult threshold validation but also vastly increases the capacity of triplet loss learning to construct expression feature representations.We find that each threshold of the triplet loss intrinsically determines a distinctive distribution of inter-class variations and corresponds, thus, to a unique expression feature representation. Therefore, rather than selecting a single optimal threshold from a valid threshold range, we thoroughly sample thresholds across the range, allowing the representation characteristics manifested by thresholds within the range to be fully extracted and leveraged for FER. To realize this approach, we partition the embedding layer of the deep metric learning network into a collection of slices and model training these embedding slices as an end-to-end multi-threshold deep metric learning problem. Each embedding slice corresponds to a sample threshold and is learned by enforcing the corresponding triplet loss, yielding a set of distinct expression features, one for each embedding slice. It makes the embedding layer, which is composed of a set of slices,a more informative and discriminative feature, hence enhancing the FER accuracy. In addition, conventional triplet loss may fail to converge when using the popular *Batch*
*Hard* strategy to mine informative triplets. We suggest that this issue is essentially a result of the so-called “cycle of incomplete judgements” inherent in the conventional triplet loss. In order to address this issue, we propose a new loss known as dual triplet loss. The new loss is simple, yet effective, and converges rapidly. Extensive evaluations demonstrate the superior performance of the proposed approach on both posed and spontaneous facial expression datasets.

## Environment

The code is developed and tested under the following environment:

- Python 3.6
- Tensorflow 1.14

## Usage
Step one, use aug_data.py in the tool folder for offline image augmentation. shape_predictor_68_face_landmarks.dat download from the link: https://pan.baidu.com/s/1NR9s7a-XO9vXgTxJOCLeiQ?pwd=te6j extraction code: te6j

Step two, use make_txt.py in the tool folder to generate a txt file containing paths to the images.

Train： Modify the path in the code according to the generated txt file path, then execute 'bash [train.sh](http://train.sh/)' or 'python train_seven_branch.py'. The pre-trained files are located in the pre_model folder.

Test： Modify the code paths according to the generated txt file path, then execute 'bash [test.sh](http://test.sh/)' or 'python [test.py](http://test.py/)'. The trained model is located in the model folder

pre_model folder and model folder can download form Baidu Netdisk  

link：https://pan.baidu.com/s/1NTMgYs4IHfFLaYGYJa_fLw?pwd=bmy5 
extraction code：bmy5

## Citations




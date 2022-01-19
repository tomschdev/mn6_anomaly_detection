# Robust Anomaly Detection in CCTV Surveillance: 
## Author: Thomas Scholtz (21681147@sun.ac.za)

### This repository hosts files that implement components of an anomaly detection framework. The framework combines inputs from three sources to provide a final verdict on the perceived degree of anomaly contained in CCTV surveillance. The first approach is an implementation of previous work in anomaly detection. The second and third are novel anomaly detection heuristics. 
### Results of evaluation on unseen footage are presented in a [web application](https://share.streamlit.io/tomschdev/cctvanomalydetection/demo/src/pred_evaluation.py) 

[c3d](https://git.cs.sun.ac.za/21681147/mn6-anomaly-detection/-/tree/master/c3d_features) contains files pertaining to the implementation of a 3D Convolutional Net. The network is used to extract C3D features (.fc6) from 200GB of CCTV footage which forms the UCF-Crime data set. The C3D module ultimately outputs a binary representation of a 4096D vector, per 16 frames of a video. 

[annMIL](https://git.cs.sun.ac.za/21681147/mn6-anomaly-detection/-/tree/master/annMIL/src/main) receives 4096D feature vectors which are representative of the data set and implements the training and testing procedures of an Artificial Neural Network wrapped in a Multiple Instance Learning framework which provides the costs to the ANN via an anomaly ranking model. Tensorflow + Keras was used to implement low-level custom functionality required for the unique combination of training schedule and provison of loss.

[heuristics](https://git.cs.sun.ac.za/21681147/mn6-anomaly-detection/-/tree/master/raft) contains the implementations of CRAFT (Consecutive frame construction with RAFT) and LKKM (Lukas-Kanade K-Means). These components translate optical flow methods into anomaly detection heuristics which perform computations on a per-frame basis for fine-grained anomaly detection.

[demo](https://git.cs.sun.ac.za/21681147/mn6-anomaly-detection/-/tree/master/demo) contains code which performs score processing and combination such that a consensus score profile is formed. Thereafter, the score profiles are evaluated against test set annotations. 

## Acknowledgements
This work was supervised by Dr M. Ngxande (ngxandem@sun.ac.za). </br>
Significant guidance has been drawn from the papers contained in [docs/reseach_papers](https://git.cs.sun.ac.za/21681147/mn6-anomaly-detection/-/tree/master/docs/guiding_papers). </br>
The following papers are particularly relevant to the implementation of the framework:
- Sultani, W., Chen, C. and Shah, M., 2018. Real-world anomaly detection in surveillance videos. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 6479-6488).
- Tran, D., Bourdev, L., Fergus, R., Torresani, L. and Paluri, M., 2015. Learning spatiotemporal features with 3d convolutional networks. In Proceedings of the IEEE international conference on computer vision (pp. 4489-4497).
- Teed, Z. and Deng, J., 2020, August. Raft: Recurrent all-pairs field transforms for optical flow. In European conference on computer vision (pp. 402-419). Springer, Cham. 

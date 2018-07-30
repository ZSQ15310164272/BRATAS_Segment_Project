# BRATAS_Segment_Project
This is an implementation for brain tumor segmentation U-net by finetuning VGG16 model.

1.Preparing Stage 
Required Library: 
SimpleITK 
numpy  
tensorflow  
keras  
matplotlib  
skimage  
h5py  
pickle 
cv2  

2.Finetue Stage 
vgg16_weights_tf_dim_ordering_tf_kernels.h5   
vgg16_weights_tf_dim_ordering_tf_kernels_notopto.h5  

3.Code Stage 
GenerateData.py to convert *.mha to *.npz for constructing trainingData small batch. 
modelTrain.py to train the unique U-net model constructed for finetuing VGG16. 
modelTest.py to test trained model. 
showResult to plot segment results.

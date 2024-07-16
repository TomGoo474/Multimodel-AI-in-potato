Multimodel AI in potato disease detection.

%Download potato dataset:
Link：https://pan.baidu.com/s/1Yi0ybJMysTbs6_BcywssDg 
Extraction code：zxcv

%Runing models:
%Step 1: Train Vision Models in detecting potato disease.
      Step 1.1: Download Vision Models' structure;
            Link：https://pan.baidu.com/s/17juj1yahiim6rT4bXMj0GA 
            Extraction code：zxcv
      Step 1.2: Run "MSC_ResViT.m".
      Trained Models, you can get these models from fellow link:
            Link：https://pan.baidu.com/s/1JHdChJX4u_vljCjGVKHcZg 
            Extraction code：zxcv
%Step 2: Train Text Model in detecting patato disease.
      Step 2.1: Load textdataset.mat
      Step 2.2: Run "ClassifyTextDataUsingConvolutionalNeuralNetwork.mlx"
%Step 3:Train CT(Color&Texture feature) CNN.
      Step 3.1: Run "Colorsfeature.mlx".
      Step 3.2: Run "TextureScattering.mlx".
      Step 3.3: Load net_CT.mat.
      Step 3.4: Run "CTCNN.mlx".
%Step 4: Combine three models' feature, and train subspace discrimination.
      Step 4.1: Load multimodel_feature.mat.
      Step 4.2: Run "combinefeature.mlx".
      Step 4.3: Use Matlab APP "classification" to train subspace discrimination.
      

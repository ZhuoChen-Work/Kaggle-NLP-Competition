# Kaggle-NLP-Competition

https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/

This Kaggle NLP competition was held in 2019, our team ranked 77th in the end. The code was written in TF v1.13 and the TPU implementation was different from current version. To reproduce the previous work, I rewrite the code of BERT part with TF2.4 and the Huggingface package.

The model can detect the toxicity of the text and reduce the bias caused by the 9 specifical topics. The following table shows the accuracy of the model on different subsets of the validation set. The detail of the metric is shown in https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview/evaluation

Subgroup AUC: Here, we restrict the data set to only the examples that mention the specific identity subgroup. A low value in this metric means the model does a poor job of distinguishing between toxic and non-toxic comments that mention the identity.

BPSN (Background Positive, Subgroup Negative) AUC: Here, we restrict the test set to the non-toxic examples that mention the identity and the toxic examples that do not. A low value in this metric means that the model confuses non-toxic examples that mention the identity with toxic examples that do not, likely meaning that the model predicts higher toxicity scores than it should for non-toxic examples mentioning the identity.

BNSP (Background Negative, Subgroup Positive) AUC: Here, we restrict the test set to the toxic examples that mention the identity and the non-toxic examples that do not. A low value here means that the model confuses toxic examples that mention the identity with non-toxic examples that do not, likely meaning that the model predicts lower toxicity scores than it should for toxic examples mentioning the identity.

![image](https://user-images.githubusercontent.com/118645613/213930738-4c6eee65-4d03-4409-933c-a5e0baecb9f4.png)

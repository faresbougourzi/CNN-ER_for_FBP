## Deep Learning Based Face Beauty Prediction via Dynamic Robust Losses and Ensemble Regression

In summary, the main contributions of this paper are as follows:

- We propose ParamSmoothL1 regression loss function. In addition, we introduce a
dynamic law that changes the parameter of the robust loss function during train-
ing. To this end, we use the cosine law with the following robust loss functions:
ParamSmoothL1, Huber and Tukey. This can solve the issue of complexity in
searching the best loss function parameter.

- We propose two branches network (REX-INCEP) for facial beauty estimation based
on ResneXt-50 and Inception-v3 architectures. The main advantage of our REX-
INCEP architecture is its ability to learn high-level FBP features using ResneXt and
Inception blocks simultaneously, which proved its efficiency compared to seven CNN
architectures. Moreover, our REX-INCEP architecture provides the right tradeoff
between the performance and the number of parameters for facial beauty prediction.

- We propose ensemble regression for facial beauty estimation by fusing the predicted
scores of one branch networks (ResneXt-50 and Inception-v3) and two branches
network (REX-INCEP) which are trained using four loss functions (MSE, dynamic
ParamSmoothL1, dynamic Huber and dynamic Tukey). Although the individual
regression models are trained using the same fixed hyper-parameters, the resulting
ensemble regression yields the best accurate estimates compared to the individual
models and to the state-of-the-art solutions.

# Damo-Embedding

This project is mainly aimed at the model training scenario of small companies, because small companies may be limited in machine resources, and it is not easy to apply for large memory machines or distributed clusters. In addition, most small companies do not need distributed training when training machine learning/deep learning models. On the one hand, because small companies do not have enough data to train distributed large models. On the other hand, training distributed model is a relatively complex project, with high requirements for engineers, and the cost of machines is also high. However, if stand-alone training is used, Out-Of-Memory (OOM) and Out-Of-Vocabulary (OOV) problems often arise. Damo-Embedding is a project designed to solve these problems.

## Out-Of-Memory(OOM)

When using the machine learning framework (TensorFlow/Pytorch) to train the model, creating a new embedding is usually necessary to specify the dimension and size in advance. And also, their implementations are based on memory. If the embedding size is too large, there will be no enough memory. So why do you need such a large Embedding? Because in some scenarios, especially in search, recommmend or ads scenarios, the number of users and materials is usually very large, and engineers will do some manual cross-features, which will lead to exponential expansion of the number of features.

## Out-Of-Vocabulary(OOV)

In the online training model, some new features often appear, such as new user ids, new material ids, etc., which have never appeared before. This will cause the problem of OOV.

## Solutions

The reason for the OOV problem is mainly because the embedding in the training framework is implemented in the form of an array. Once the feature id is out of range, the problem of OOV will appear. We use [rocksdb](https://rocksdb.org/) to store embedding, which naturally avoids the problems of OOV and OOM, because rocksdb uses KV storage, which is similar to hash table and its capacity is only limited by the local disk.

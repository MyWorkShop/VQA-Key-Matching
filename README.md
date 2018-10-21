# VQA-Key-Matching

## What is Visual Question Answering?

VQA is a new dataset containing open-ended questions about images. These questions require an understanding of vision, language and commonsense knowledge to answer.[1]

## What is Key Matching?

Traditional method uses neural networks to conpute the hotmap.[2]

Inspired by the key addressing in the Neural Turing Machine[3]. We compute our hot map using key matching.

The convolution neural networks for image will output a key map and a feature map, and the convolution neural networks for question will output a feature vecotr and a key vector. The hot map will be computed by a softmax function on the cosine similarity between each key on the key map and the key vector.

## How to use these code?

We use VQA v2 dataset and you can download it [here](http://visualqa.org/download.html).

The annotations file, the images file and the questions file are required separately to be put into `Annotations`, `Images` and `Questions` floder.

Your are reconmanded to run `tr_read.py` to load the image file in the datasets to a `.tfrecord` file. The you can just run `train.py` to train the model.

## Why there is no test code?

This project is still under construction. We're still optimizing the model on the training set. We're sure test code will be produced in the future.

## Reference
[1]: [VQA: Visual Question Answering](http://visualqa.org/)

[2]: Yang, Zichao, et al. "Stacked attention networks for image question answering." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.

[3]: Graves, Alex, Greg Wayne, and Ivo Danihelka. "Neural turing machines." arXiv preprint arXiv:1410.5401 (2014).

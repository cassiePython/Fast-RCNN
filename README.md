# Implementation of Fast-RCNN

This is a Implementation of Fast-RCNN. 

## Prerequisites
- Python 2.7/3.5
- Pytorch 0.3.1
- cv2 3.4.0

You can run the code in Windows/Linux with CPU/GPU. 

## Dataset

For simplicity, I use the Vehicle Datase of Beijing Institute of Technology for trainging and testing. It can be downloaded from Baidu Drive: 

https://pan.baidu.com/s/1X-8E5eGldAfTHdyJXlFllA
Passward: ivq8

## Structure

The project is structured as follows:

```
├── checkpoints/
├── data/
|   ├── dataset_factory.py    
|   ├── datasets.py    
├── generate/
├── loss/
|   ├── losses.py  
├── models/
|   ├── model_factory.py    
|   ├── models.py  
├── networks/
|   ├── network_factory.py    
|   ├── networks.py 
├── options/
|   ├── base_options.py    
|   ├── test_options.py 
|   ├── train_options.py
├── sample_dataset/
|   ├── Annotations   
|   ├── Images 
│   ├── test_list.txt
|   ├── train_list.txt
├── utils/
|   ├── selectivesearch.py    
|   ├── util.py 
├── evaluate.py
├── train.py
```

## Getting started

### Supervised Train

Use pre-trained AlexNet of Pytorch and train it using the Vehicle Datase. 

```
$ python train.py 
```

You can directly run it with default parameters.

### Evaluate

```
$ python evaluate.py --load_epoch 20 --img_path ./sample_dataset/Images/000032.jpg
```

![](https://github.com/bigbrother33/Deep-Learning/blob/master/photo/1.PNG)

## References

- Selective-search: https://github.com/AlpacaDB/selectivesearch
- Fast-RCNN with Tensorflow: https://github.com/Liu-Yicheng/Fast-RCNN

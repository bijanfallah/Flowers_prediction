# Flowers' Species Prediction

A code to distinguish the species of a flower image. There are 102 species recorded in the [dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html). 
The model has 80% accuracy on test data! 

![Flowers](https://github.com/bijanfallah/Flowers_prediction/blob/master/index.png)


# How to train the model? 
`python train.py data-directory --args`
### example: 
`python train.py --gpu "cpu" --data_dir 'flowers'`

- Arguments: 
* --data_dir   data directory
* --save_dir   saving directory
* --arch       torch model architecture : vgg19, vgg16, ResNet-101, ResNet-151
* --learning_rate learning rate 
* --gpu        gpu or cpu
* --top_k      top k predictions
* --chpo       checkpoint file 
* --hidden_units number of hidden units
* --epochs     number of epochs

# How to conduct predictions? 
`python predict.py --gpu "cpu"`

- 

# Week1 programming assignment - TinyImageNet200

  

## Environment settings

### Pre-requisite
Before we begin, the host server should meet the requirements below:
* Docker >= 19.03.8
* CUDA >= 10.2

###  Download the data

```
cd data
sh download_and_unzip.sh # The data are saved at ./data/tiny-imagenet-100
```

### Build and run the docker image

```
cd Docker
bash build_docker.sh # Build the docker image
sh run_docker.sh # create a docker container
docker attach <DOCKER_CONTAINER_NAME> # attach to the docker container
```

## Task

### Task 0: experiment logging
* use [wandb](https://www.wandb.com/) for logging the training and validation process.
* The metrics below should be included:
	* training loss
	* training accuracy (top-1 top-5)
	* validation accuracy (top-1 top-5)
	* hard sample images (training / validation)
	* confusion matrix
* Various types of data can be logged to wandb. Refer to [the documentation](https://docs.wandb.com/library/log)

### Task 1: hyper-parameter tuning

* Tune the hyper-params
	* learning rate, batch size, optimizer, learning rate scheme (including number of epochs)
	* may change the backbone architecture (default resnet18)

### Task 2: augmentations

* Add augmentations for training and validation.
	* [PyTorch augmentation library](https://pytorch.org/docs/stable/torchvision/transforms.html)
	* [imgaug](https://github.com/aleju/imgaug)
	* [autoaugment](https://github.com/DeepVoltaire/AutoAugment)
* When adding augmentations for validation, we ensemble multiple 'views' of the samples.
* Please do not slow down the training time due to the augmentations.

### Task 3: let's try our best

* The top entries in cs213n in-class challenge have 17~20% errors in the test set. Let's try to achieve that performance in the validation set.
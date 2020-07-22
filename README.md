# View from the Neural Networks

This program generates video files visualized by the AlexNet/VGG19 architectures.

## Usage
```
$ python nnview.py <input-video-file> <output-video-file>
```

## Genarated videos
* `nnview.py` generates videos of all feature maps from [the pigeons' video](https://pixabay.com/videos/id-39264/).

### AlexNet
![AlexNet](images/nnview-pigeons-alexnet.gif)

### VGG19
![VGG19](images/nnview-pigeons-vgg19.gif)

## References
  * https://github.com/kkroening/ffmpeg-python
  * https://pytorch.org/hub/pytorch_vision_alexnet/
  * https://pytorch.org/hub/pytorch_vision_vgg/

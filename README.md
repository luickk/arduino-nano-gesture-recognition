# Food Watch

The Food Watch is a strap (or a watch) which alerts the user when he eats something (via gesture recognition). The strap consists of an Arduino Nano 33 sense which utilizes a tensorflow lite model, which is trained to recognize gestures associated with eating. The training data consists of 3 vector IMU accelerometer and gyro data, with a total of 6 features.

## Neural Net

### RNN

Since a gesture resembles a sequence of data with a special pattern in a continuous data flow, the usage of a recurrent neural net seems like a adequat solution. But since RNNs are not ment to be used for a continous data flow and the sliding window technique it would neither be efficient nor useful.
Additionally the by the RNN required tensorflow operation that are not supported by tensorflow lite, which is needed make the model run on a microcontroller or mobile phone.

### CNN

A CNN might not be ideal for squential data recognition but is perfect to be used as a sliding window and, continous sequenced data flows.
You can fin further information on that topic [here](https://medium.com/@jon.froiland/convolutional-neural-networks-for-sequence-processing-part-1-420dd9b500).

## TFLite conversion

### Convert keras model to tflite model

To actually run the keras model on the Arduino Nano you need to convert it to an tflite model. To do that, run
`bash to_tf_lite.sh <keras model path>` and replace the model input/ output name.

### Convert tflite model to c array

For that xxd is required, which is preinstalld on most linux distros.

`xxd -i converted_model.tflite > model_data.cc`

## Run model on arduino

Resources for that can be found [here](https://www.tensorflow.org/lite/microcontrollers/library)


# Depndencies

- keras v2.1.5

- tensorflow-gpu v2.0 <br>

- CUDA Toolkit v9.0

- cuDNN v7.6.5

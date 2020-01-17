# Food Watch

The Food Watch is a strap (or a watch) which alerts the user when he eats something (via gesture recognition). The strap consists of an Arduino Nano 33 sense which utilizes a tensorflow lite model, which is trained to recognize gestures associated with eating. The training data consists of 3 vector IMU accelerometer and gyro data, with a total of 6 features.

## Setup

### Data collection and Training

First you need to collect enough data, you can do that with the `arduino/data_recording` sketch. The serial output is shown in the terminal. To save it just copy and paste it to a file. If you're collecting a lot of data, over a longer period of time you can directly pipe the output of the serial port into a file(best done on linux).

With the collected data you can then start to train your model via. `train.py`.
If you've trained a model with good enough accuracy, you then have to convert it to a tflite format and next to a c array, so that it can be used on the arduino.

### Convert keras model to tflite model/ c array

To actually run the keras model on the Arduino Nano you need to convert it to an tflite model. To do that, run
`bash to_tf_lite.sh <keras model path>` and replace the model input/ output name.

To convert the tflite model to a c array, xxd is required, which is preinstalld on most linux distros.

`xxd -i converted_model.tflite > model_data.cc`

### Model deployment on arduino

To run your model on the arduino, move the generated c array model to the `aruino/tflite_classification` sketch and compile&upload the sketch.

Further resources can be found [here](https://www.tensorflow.org/lite/microcontrollers/library)

### Memory limits

If you encounter an exception that are similar to this: `ARDUINO_NANO33BLE linker_script.ld:138 cannot move location counter backwards`, it is propably due to the arduino running out of memory. In this case you will have to shrink your trained model in size. Side note, the Arduino Nano Sense 33 has a total of 256KB, so your model should not exceed that. As a rough orientation you can substract the size of the source code and take that as a model size limit.  

## Neural Net Choice

### RNN

Since a gesture resembles a sequence of data with a special pattern in a continuous data flow, the usage of a recurrent neural net seems like a adequat solution. But since RNNs are not ment to be used for a continous data flow and the sliding window technique it would neither be efficient nor useful.
Additionally the by the RNN required tensorflow operation that are not supported by tensorflow lite, which is needed make the model run on a microcontroller or mobile phone.

### CNN

A CNN might not be ideal for squential data recognition but is perfect to be used as a sliding window and, continous sequenced data flows.
You can fin further information on that topic [here](https://medium.com/@jon.froiland/convolutional-neural-networks-for-sequence-processing-part-1-420dd9b500).

# Depndencies

- keras v2.1.5

- tensorflow-gpu v2.0 <br>

- CUDA Toolkit v9.0

- cuDNN v7.6.5

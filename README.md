# Food Watch

The food watch is a strap (or a watch) which alerts the user when he eats something (via gesture recog.). The strap consists of an arduino nano 33 sense which utalizes a tensorflow lite model, which is trained to recognize gestures associated with eating. The training data consists of 3 vector IMU accelerometer and gyro data.

## Neural Net

### RNN

Since a gesture resembles a sequence of data with a special pattern in a continuous data flow, the usage of a recurrent neural net seems like a adequat solution. But since RNNs are not ment to be used for a continous data flow and the sliding window technique it would neither be efficient nor useful.
Additionally the by the RNN required tensorflow operation that are not supported by tensorflow lite, which is needed make the model run on a microcontroller or mobile phone.

# Depndencies

- keras v2.1.5

- tensorflow-gpu v1.8.0 <br>

- CUDA Toolkit v9.0

- cuDNN v7.6.5

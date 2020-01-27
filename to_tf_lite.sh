tflite_convert --keras_model_file=$1 --output_file=conv.tflite --inference_type=QUANTIZED_UINT8;
rm arduino/tflite_classification/live_conv_test.cc;
xxd -i conv.tflite > arduino/tflite_classification/live_conv_test.cc;
rm conv.tflite;

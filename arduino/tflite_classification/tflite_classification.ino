#include <Arduino_LSM9DS1.h> //Include the library for 9-axis IMU


#include <TensorFlowLite.h>
#include <tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h>
#include <tensorflow/lite/experimental/micro/micro_error_reporter.h>
#include <tensorflow/lite/experimental/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include "/Users/luickklippel/Documents/projekte/food-watch/arduino/tflite_classification/test_model.cc"



#include "/Users/luickklippel/Documents/projekte/food-watch/arduino/tflite_classification/live_conv_test.cc"

// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::ops::micro::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize];


// array to map gesture index to a name
const char* gestures_array[] = {"punch"};

int n_samples = 0;

int num_gests = sizeof(gestures_array) / sizeof(gestures_array[0]);

int batch_size = 40;

void setup() {
  Serial.begin(9600); //Serial monitor to display all sensor values 

  if (!IMU.begin()) //Initialize IMU sensor 
  { Serial.println("Failed to initialize IMU!"); while (1);}

  
  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(conv_tflite);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }
  pinMode(LED_BUILTIN, OUTPUT);
  
  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);
  
  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();
  
  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

float accel_x, accel_y, accel_z;
float gyro_x, gyro_y, gyro_z;

// normalized to 0-1
float naccel_x, naccel_y, naccel_z;
float ngyro_x, ngyro_y, ngyro_z;


// float mapping function ported from arduino c++ code
// cuts off add given mask value to achieve higher sensitivity
float masked_mapf(float val, float mask_min, float mask_max, float in_min, float in_max, float out_min, float out_max)
{
  float final_val;
  if(val >= mask_min and val <= mask_max) 
  {
    final_val = (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
  } else if (val < mask_min)
  {
    final_val = mask_min;
  } else if(val > mask_max)
  {
    final_val = mask_max;
  }
  return final_val;
}


// float mapping function ported from arduino c++ code
float mapf(float val, float in_min, float in_max, float out_min, float out_max) 
{
    return (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

void loop() 
{  
  //Accelerometer&Gyro values 
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) 
  {
    IMU.readAcceleration(accel_x, accel_y, accel_z);
    IMU.readGyroscope(gyro_x, gyro_y, gyro_z);
    
    // normalize the IMU data between 0 to 1 and store in the model's
    // input tensor
    naccel_x = masked_mapf(accel_x, -1, 1, -1, 1, 0, 1);
    naccel_y = masked_mapf(accel_y, -1, 1, -1, 1, 0, 1);
    naccel_z = masked_mapf(accel_z, -1, 1, -1, 1, 0, 1);
    ngyro_x = masked_mapf(gyro_x, -1000, 1000, -1000, 1000, 0, 1);
    ngyro_y = masked_mapf(gyro_y, -1000, 1000,  -1000, 1000, 0, 1);
    ngyro_z = masked_mapf(gyro_z, -1000, 1000, -1000, 1000, 0, 1);
    
    tflInputTensor->data.f[n_samples * 6 + 0] = naccel_x;
    tflInputTensor->data.f[n_samples * 6 + 1] = naccel_y;
    tflInputTensor->data.f[n_samples * 6 + 2] = naccel_z;
    tflInputTensor->data.f[n_samples * 6 + 3] = ngyro_x;
    tflInputTensor->data.f[n_samples * 6 + 4] = ngyro_y;
    tflInputTensor->data.f[n_samples * 6 + 5] = ngyro_z;
    
    n_samples++;

    // Serial.print("naX: " ); Serial.print(naccel_x);Serial.print(", naY: " ); Serial.print(naccel_y);Serial.print(", naZ: " ); Serial.println(naccel_z);
    // Serial.print("ngX: " ); Serial.print(ngyro_x);Serial.print(", ngY: " ); Serial.print(ngyro_y);Serial.print(", ngZ: " ); Serial.println(ngyro_z);
    if (n_samples == batch_size) {
      
      // Run inferencing
      TfLiteStatus invokeStatus = tflInterpreter->Invoke();
      if (invokeStatus != kTfLiteOk) {
        Serial.println("Invoke failed!");
        while (1);
        return;
      }
      n_samples = 0;
      
      // Loop through the output tensor values from the model
      for (int i = 0; i < num_gests; i++) {
        Serial.print(gestures_array[i]);
        Serial.print(": ");
        Serial.println(tflOutputTensor->data.f[i], 6);
        // Serial.print("naX: " ); Serial.print(naccel_x);Serial.print(", naY: " ); Serial.print(naccel_y);Serial.print(", naZ: " ); Serial.println(naccel_z);
        // Serial.print("ngX: " ); Serial.print(ngyro_x);Serial.print(", ngY: " ); Serial.print(ngyro_y);Serial.print(", ngZ: " ); Serial.println(ngyro_z);
      }
    }
  }
}

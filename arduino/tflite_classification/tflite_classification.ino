#include <Arduino_LSM9DS1.h> //Include the library for 9-axis IMU
#include <Arduino_LPS22HB.h> //Include library to read Pressure 
#include <Arduino_HTS221.h> //Include library to read Temperature and Humidity 
#include <Arduino_APDS9960.h> //Include library for colour, proximity and gesture recognition

#include <TensorFlowLite.h>
#include <tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h>
#include <tensorflow/lite/experimental/micro/micro_error_reporter.h>
#include <tensorflow/lite/experimental/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include "C:/Users/gelbe/Documents/Projekte/food-watch/arduino/tflite_classification/222_model.cc"
#include "C:/Users/gelbe/Documents/Projekte/food-watch/arduino/tflite_classification/test_model.cc"

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

  if (!BARO.begin()) //Initialize Pressure sensor 
  { Serial.println("Failed to initialize Pressure Sensor!"); while (1);}

  if (!HTS.begin()) //Initialize Temperature and Humidity sensor 
  { Serial.println("Failed to initialize Temperature and Humidity Sensor!"); while (1);}

  if (!APDS.begin()) //Initialize Colour, Proximity and Gesture sensor 
  { Serial.println("Failed to initialize Colour, Proximity and Gesture Sensor!"); while (1);}

  
  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(test_model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }
  
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

void loop() 
{  
  //Accelerometer values 
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) 
  {
    IMU.readAcceleration(accel_x, accel_y, accel_z);
    IMU.readGyroscope(gyro_x, gyro_y, gyro_z);
    
    // normalize the IMU data between 0 to 1 and store in the model's
    // input tensor
    tflInputTensor->data.f[n_samples * 6 + 0] = (accel_x + 4.0) / 8.0;
    tflInputTensor->data.f[n_samples * 6 + 1] = (accel_y + 4.0) / 8.0;
    tflInputTensor->data.f[n_samples * 6 + 2] = (accel_z + 4.0) / 8.0;
    tflInputTensor->data.f[n_samples * 6 + 3] = (gyro_x + 2000.0) / 4000.0;
    tflInputTensor->data.f[n_samples * 6 + 4] = (gyro_y + 2000.0) / 4000.0;
    tflInputTensor->data.f[n_samples * 6 + 5] = (gyro_z + 2000.0) / 4000.0;
    
    n_samples++;
  
    if (n_samples == batch_size) {
      // Run inferencing
      TfLiteStatus invokeStatus = tflInterpreter->Invoke();
      if (invokeStatus != kTfLiteOk) {
        Serial.println("Invoke failed!");
        while (1);
        return;
      }
    
      // Loop through the output tensor values from the model
      for (int i = 0; i < num_gests; i++) {
        Serial.print(gestures_array[i]);
        Serial.print(": ");
        Serial.println(tflOutputTensor->data.f[i], 6);
      }
      n_samples = 0;
    }
  }
}

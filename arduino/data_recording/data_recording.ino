#include <Arduino_LSM9DS1.h> //Include the library for 9-axis IMU
#include <Arduino_LPS22HB.h> //Include library to read Pressure 
#include <Arduino_HTS221.h> //Include library to read Temperature and Humidity 
#include <Arduino_APDS9960.h> //Include library for colour, proximity and gesture recognition

int eating_identifier_button = 12;

void setup()
{
  Serial.begin(9600); //Serial monitor to display all sensor values 

  if (!IMU.begin()) //Initialize IMU sensor 
  { Serial.println("Failed to initialize IMU!"); while (1);}

  if (!BARO.begin()) //Initialize Pressure sensor 
  { Serial.println("Failed to initialize Pressure Sensor!"); while (1);}

  if (!HTS.begin()) //Initialize Temperature and Humidity sensor 
  { Serial.println("Failed to initialize Temperature and Humidity Sensor!"); while (1);}

  if (!APDS.begin()) //Initialize Colour, Proximity and Gesture sensor 
  { Serial.println("Failed to initialize Colour, Proximity and Gesture Sensor!"); while (1);}

  
  Serial.print("gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z");

  pinMode(eating_identifier_button, INPUT); 
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

    if (digitalRead(eating_identifier_button) == HIGH) 
    {
      Serial.print(gyro_x); Serial.print(",");Serial.print(gyro_y);Serial.print(",");Serial.print(gyro_z);Serial.print(",");Serial.print(accel_x); Serial.print(",");Serial.print(accel_y);Serial.print(",");Serial.print(accel_z);Serial.println(",1");
     
    } else 
    {
      Serial.print(gyro_x); Serial.print(",");Serial.print(gyro_y);Serial.print(",");Serial.print(gyro_z);Serial.print(",");Serial.print(accel_x); Serial.print(",");Serial.print(accel_y);Serial.print(",");Serial.print(accel_z);Serial.println(",0");
     
    }
    
    
    // Serial.print("Gyroscope = ");Serial.print(gyro_x); Serial.print(", ");Serial.print(gyro_y);Serial.print(", ");Serial.println(gyro_z);
    // Serial.print("Accelerometer = ");Serial.print(accel_x); Serial.print(", ");Serial.print(accel_y);Serial.print(", ");Serial.println(accel_z);
  }
}

#include <Wire.h>

// Placeholder Arduino noise capture sketch.
// Replace with your exact sensor wiring and libraries.

void setup() {
  Serial.begin(115200);
}

void loop() {
  unsigned long t = millis();

  // Replace with actual sensor reads:
  float temp_noise = (analogRead(A0) / 1023.0) - 0.5;
  float imu_noise = (analogRead(A1) / 1023.0) - 0.5;
  float voltage_noise = (analogRead(A2) / 1023.0) - 0.5;

  Serial.print(t);
  Serial.print(",");
  Serial.print(temp_noise, 6);
  Serial.print(",");
  Serial.print(imu_noise, 6);
  Serial.print(",");
  Serial.println(voltage_noise, 6);

  delay(100); // 10 Hz
}

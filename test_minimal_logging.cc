/* Test program to demonstrate the modified logging functionality */
#include <iostream>
#include "tensorflow/lite/minimal_logging.h"

int main() {
  std::cout << "Testing TensorFlow Lite Minimal Logging with modified severity names...\n\n";

  // Set minimum log severity to VERBOSE to show all messages
  tflite::logging_internal::MinimalLogger::SetMinimumLogSeverity(TFLITE_LOG_VERBOSE);

  // Test different log levels
  TFLITE_LOG_PROD(TFLITE_LOG_VERBOSE, "This is a verbose message with number: %d", 42);
  TFLITE_LOG_PROD(TFLITE_LOG_INFO, "This is an info message: %s", "Hello TensorFlow!");
  TFLITE_LOG_PROD(TFLITE_LOG_WARNING, "This is a warning message: value=%f", 3.14159);
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "This is an error message: code=%d", 500);

  std::cout << "\nNow testing with minimum severity set to WARNING...\n\n";
  
  // Change minimum severity to WARNING
  tflite::logging_internal::MinimalLogger::SetMinimumLogSeverity(TFLITE_LOG_WARNING);
  
  // These should not appear
  TFLITE_LOG_PROD(TFLITE_LOG_VERBOSE, "This verbose message should NOT appear");
  TFLITE_LOG_PROD(TFLITE_LOG_INFO, "This info message should NOT appear");
  
  // These should appear
  TFLITE_LOG_PROD(TFLITE_LOG_WARNING, "This warning SHOULD appear");
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "This error SHOULD appear");

  std::cout << "\nTest completed successfully!\n";
  return 0;
}
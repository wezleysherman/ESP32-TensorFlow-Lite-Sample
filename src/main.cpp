/*=============================================================================
TensorFlow Lite Platformio Example

Author: Wezley Sherman
Referenced Authors: The TensorFlow Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <Arduino.h>
#include <math.h>
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "sine_model_data.h"

// Create a memory pool for the nodes in the network
constexpr int tensor_pool_size = 2 * 1024;
uint8_t tensor_pool[tensor_pool_size];

// Define the model to be used
const tflite::Model* sine_model;

// Define the interpreter
tflite::MicroInterpreter* interpreter;

// Input/Output nodes for the network
TfLiteTensor* input;
TfLiteTensor* output;

// Set up the ESP32's environment.
void setup() { 
	// Start serial at 115200 baud
	Serial.begin(115200);

	// Load the sample sine model
	Serial.println("Loading Tensorflow model....");
	sine_model = tflite::GetModel(g_sine_model_data);
	Serial.println("Sine model loaded!");

	// Define ops resolver and error reporting
	static tflite::ops::micro::AllOpsResolver resolver;

	static tflite::ErrorReporter* error_reporter;
	static tflite::MicroErrorReporter micro_error;
	error_reporter = &micro_error;

	// Instantiate the interpreter 
	static tflite::MicroInterpreter static_interpreter(
		sine_model, resolver, tensor_pool, tensor_pool_size, error_reporter
	);

	interpreter = &static_interpreter;

	// Allocate the the model's tensors in the memory pool that was created.
	Serial.println("Allocating tensors to memory pool");
	if(interpreter->AllocateTensors() != kTfLiteOk) {
		Serial.println("There was an error allocating the memory...ooof");
		return;
	}

	// Define input and output nodes
	input = interpreter->input(0);
	output = interpreter->output(0);
	Serial.println("Starting inferences... Input a number! ");
}

// Logic loop for taking user input and outputting the sine
void loop() { 
	// Wait for serial input to be made available and parse it as a float
	if(Serial.available() > 0) {
    	float user_input = Serial.parseFloat();

    	/* The sample model is only trained for values between 0 and 2*PI
    	 * This will keep the user from inputting bad numbers. 
    	 */
		if(user_input < 0.0f || user_input > (float)(2*M_PI)) {
			Serial.println("Your number must be greater than 0 and less than 2*PI");
			return;
		}
    	
    	// Set the input node to the user input
    	input->data.f[0] = user_input;

    	Serial.println("Running inference on inputted data...");

    	// Run inference on the input data
    	if(interpreter->Invoke() != kTfLiteOk) {
    		Serial.println("There was an error invoking the interpreter!");
    		return;
    	}

    	// Print the output of the model.
    	Serial.print("Input: ");
    	Serial.println(user_input);
    	Serial.print("Output: ");
    	Serial.println(output->data.f[0]);
    	Serial.println("");

    }
} 
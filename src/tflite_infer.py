import time
from pathlib import Path
import numpy as np
import tflite_runtime.interpreter as tflite
from dataset import make_dataset



# Load the TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path="qat/qat_tfliteconvert.tflite", 
                                 experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()


print(interpreter.get_input_details())
print(interpreter.get_output_details())

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]


valid_x, valid_y1, valid_y2 = make_dataset(range(600, 700))

for i in range(200):
    test_image = np.expand_dims(valid_x[i], axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, test_image)

    start_time = time.monotonic()
    interpreter.invoke()
    stop_time = time.monotonic()

    output = interpreter.tensor(output_index)

    print(i, stop_time - start_time)



# outputs = output()[0]


# ssims = outputs[:64]
# sizes = outputs[64:]

# print(ssims)
# print(sizes)

# print(valid_y1[65])
# print(valid_y2[65])


# for i in range(64):
#     print(f"SSIM: {ssims[i]:.6f} {valid_y1[65][i]:.6f} SIZE: {sizes[i]:.6f} {valid_y2[65][i]:.6f}")


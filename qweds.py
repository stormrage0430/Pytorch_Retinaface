import cv2
import torch
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import onnx


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def build():
    onnx_model = onnx.load('FaceDetector.onnx')
    onnx.checker.check_model(onnx_model)
    onnx_file_path = 'FaceDetector.onnx'
    # a = cv2.dnn.readNetFromONNX('FaceDetector.onnx')
    n_channel, n_height, n_width = 3, 640, 640
    dimensions = [n_channel, n_height, n_width]
    batch_size = 1
    precision = 'fp32'

    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

    # Create builder, network, and parser
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Configure the builder here.
    builder.max_workspace_size = 2**30

    with open(onnx_file_path, 'rb') as model:
        parser.parse(model.read())
    network.mark_output(network.get_layer(network.num_layers-1).get_output(0))
    engine = builder.build_cuda_engine(network)

    engine_file_name = 'FaceDetector.trt'
    engine_file_name = engine_file_name.format('t4', batch_size, precision)

    with open(engine_file_name, 'wb') as file:
        print('Saving engine file to:', engine_file_name)
        file.write(engine.serialize())

def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)  # INFO
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def build_engine(onnx_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)   # INFO
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        if builder.platform_has_fast_fp16:
            print('this card support fp16')
        if builder.platform_has_fast_int8:
            print('this card support int8')

        builder.max_workspace_size = 1 << 30
        with open(onnx_file_path, 'rb') as model:
           parser.parse(model.read())
        return builder.build_cuda_engine(network)

def save_engine(engine, engine_dest_path):
    buf = engine.serialize()
    with open(engine_dest_path, 'wb') as f:
        f.write(buf)

# ennn = build_engine('FaceDetector.onnx')
# save_engine(ennn, 'asddd.trt')

engine_file_path = 'asddd.trt'
img_numpy = np.random.randn(1,3,640,640).astype(np.float32)
with load_engine(engine_file_path) as engine, \
        engine.create_execution_context() as context:
    ''' 3.2 - 分配host，device端的buffer'''
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    print('Running inference on image {}...')

    ''' 3.3 - 进行inference'''
    inputs[0].host = img_numpy
    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

print(trt_outputs)
print(trt_outputs.size())
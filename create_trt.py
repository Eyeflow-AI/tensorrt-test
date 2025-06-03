import os
import numpy as np

import onnx
import onnx_graphsurgeon as gs

import tensorrt as trt
#----------------------------------------------------------------------------------------------------------------------------------

def create_pad_onnx(onnx_file_name):
    opset_version = 15
    data_directory_path = "./"
    onnx_file_path = os.path.join(data_directory_path, onnx_file_name)

    input_shape = (1, gs.Tensor.DYNAMIC, gs.Tensor.DYNAMIC, gs.Tensor.DYNAMIC)
    input_image = gs.Variable(name="input_image", dtype=np.float32, shape=input_shape)
    pad_size_tensor = gs.Variable(name="pad_sizes", dtype=np.int64, shape=[8])
    paded_image = gs.Variable(name="paded_image", dtype=np.float32, shape=input_shape)

    nodes = [
        gs.Node(name="Pad-Image",
                op="Pad",
                inputs=[input_image, pad_size_tensor],
                outputs=[paded_image],
                attrs={}
        )
    ]

    graph = gs.Graph(nodes=nodes,
                     inputs=[input_image, pad_size_tensor],
                     outputs=[paded_image],
                     opset=opset_version)
    model = gs.export_onnx(graph)
    onnx.save(model, onnx_file_path)

    print(f"ONNX file created and saved: {onnx_file_path}")
# ---------------------------------------------------------------------------------------------------------------------------------

def convert_model_trt(onnx_file, trt_file):
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    onnx_file_date = os.path.getmtime(onnx_file)
    if os.path.isfile(trt_file):
        if onnx_file_date < os.path.getmtime(trt_file):
            return

    print(f"Converting Model {onnx_file} to TensorRT - Version: {trt.__version__}")
    trt_logger = trt.Logger(min_severity=trt.Logger.INFO)

    builder = trt.Builder(trt_logger)
    config = builder.create_builder_config()

    network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)

    print("parsing model")
    onnx_path = os.path.realpath(onnx_file)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print(f"Failed to load ONNX file: {onnx_path}")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise

    print("model parsed")
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    print("Network Description")
    for input in inputs:
        print(f"Input '{input.name}' with shape {input.shape} and dtype {input.dtype}")
    for output in outputs:
        print(f"Output '{output.name}' with shape {output.shape} and dtype {output.dtype}")

    trt_file = os.path.realpath(trt_file)
    engine_dir = os.path.dirname(trt_file)
    os.makedirs(engine_dir, exist_ok=True)

    profile = builder.create_optimization_profile()
    for input in inputs:
        if input.name == "input_image":
            profile.set_shape(
                input.name,
                ([1, 320, 320, 1]),
                ([1, 1280, 1920, 1]),
                ([1, 5000, 5000, 3])
            )
            print(f"Profile for Input '{input.name}' shape {input.shape}")
        elif input.name == "pad_sizes":
            profile.set_shape_input(
                input.name,
                ([0,0,0,0,0,0,0,0]),
                ([0,0,0,0,0,0,0,0]),
                ([0,160,160,0,0,160,160,0])
            )
            print(f"Profile for Input '{input.name}' shape {input.shape}")

    config.add_optimization_profile(profile)

    print(f"Serializing engine to file: {trt_file}")
    serialized_engine = builder.build_serialized_network(network, config)
    with open(trt_file, "wb") as f:
        f.write(serialized_engine)
# ---------------------------------------------------------------------------------------------------------------------------------

create_pad_onnx("pad_layer.onnx")
convert_model_trt("pad_layer.onnx", "pad_layer.trt")

#!/usr/bin/python3

# from codecs import encode
import os
import sys
import json
import datetime
import numpy as np

from eyeflow_sdk.log_obj import CONFIG, log

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import onnx
import onnx_graphsurgeon as gs
import ctypes

import tensorrt as trt
#----------------------------------------------------------------------------------------------------------------------------------


#!/usr/bin/python3

# from codecs import encode
import os
import sys
import json
import datetime
import numpy as np

from eyeflow_sdk.log_obj import CONFIG, log

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt
#----------------------------------------------------------------------------------------------------------------------------------


def create_NMS_engine(model_props, engine_path, output_boxes, output_classes, precision):
    trt_logger = trt.Logger(min_severity=trt.Logger.INFO)

    # NMS Inputs.
    nms_input_classification = gs.Variable(
        name="input_classification",
        dtype=np.float32,
        shape=[-1, -1, output_classes]
    )

    nms_input_boxes = gs.Variable(
        name="input_boxes",
        dtype=np.float32,
        shape=[-1, -1, 4]
    )

    # NMS Outputs.
    nms_output_num_detections = gs.Variable(
        name="num_detections",
        dtype=np.int32,
        shape=[-1, 1]
    )

    nms_output_boxes = gs.Variable(
        name="detection_boxes",
        dtype=np.float32,
        shape=[-1, model_props["max_boxes"], 4]
    )

    nms_output_scores = gs.Variable(
        name="detection_scores",
        dtype=np.float32,
        shape=[-1, model_props["max_boxes"]]
    )

    nms_output_classes = gs.Variable(
        name="detection_classes",
        dtype=np.int32,
        shape=[-1, model_props["max_boxes"]]
    )

    nms_inputs = [nms_input_boxes, nms_input_classification]
    nms_outputs = [nms_output_num_detections, nms_output_boxes, nms_output_scores, nms_output_classes]

    # Plugin.
    nms_node = gs.Node(
        op="EfficientNMS_TRT",
        name="nms/non_maximum_suppression",
        inputs=nms_inputs,
        outputs=nms_outputs,
        attrs={
            'plugin_version': "1",
            'background_class': -1,
            'max_output_boxes': int(model_props["max_boxes"]),
            'score_threshold': float(model_props["confidence_threshold"]),
            'iou_threshold': float(model_props["nms_iou_threshold"]),
            'score_activation': False,
            'box_coding': 0
        }
    )

    graph_nms = gs.Graph(nodes=[nms_node], inputs=nms_inputs, outputs=nms_outputs, opset=13)
    graph_nms.cleanup().toposort()
    onnx_nms_model = gs.export_onnx(graph_nms)
    log.info("Created 'nms/non_maximum_suppression' NMS plugin")

    trt.init_libnvinfer_plugins(trt_logger, namespace="")

    builder = trt.Builder(trt_logger)
    config = builder.create_builder_config()

    network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)

    if not parser.parse(onnx_nms_model.SerializeToString()):
        log.error("Failed to parse ONNX NMS model")
        for error in range(parser.num_errors):
            log.error(parser.get_error(error))
        sys.exit(1)

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    log.info("Network Description")
    for input in inputs:
        log.info("Input '{}' with shape {} and dtype {}".format(input.name, input.shape, input.dtype))
    for output in outputs:
        log.info("Output '{}' with shape {} and dtype {}".format(output.name, output.shape, output.dtype))

    engine_path = os.path.realpath(engine_path)
    engine_dir = os.path.dirname(engine_path)
    os.makedirs(engine_dir, exist_ok=True)
    log.info(f"Building {precision} Engine in {engine_path}")

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            log.warning("FP16 is not supported natively on this platform/device")
        else:
            config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        raise Exception("INT8 is not supported natively on this platform/device")

    # builder.max_batch_size = 1 # min(128, int(model_props["max_num_patches_frame"]))
    max_output_boxes = int(int(model_props["max_num_patches_frame"]) * 0.75) * output_boxes
    profile = builder.create_optimization_profile()

    profile.set_shape('input_boxes', (1, output_boxes, 4), (1, max_output_boxes // 2, 4), (1, max_output_boxes, 4))
    profile.set_shape('input_classification', (1, output_boxes, output_classes), (1, max_output_boxes // 2, output_classes), (1, max_output_boxes, output_classes))

    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_path, "wb") as f:
        log.info("Serializing engine to file: {:}".format(engine_path))
        f.write(serialized_engine)

    return
# ---------------------------------------------------------------------------------------------------------------------------------


def convert_model_trt_old(onnx_file, trt_file, component_options, precision):
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    onnx_file_date = os.path.getmtime(onnx_file)
    if os.path.isfile(trt_file):
        if onnx_file_date < os.path.getmtime(trt_file):
            return

    log.info(f"Converting Model {onnx_file} to TensorRT - Version: {trt.__version__}")
    trt_logger = trt.Logger(min_severity=trt.Logger.INFO)

    trt.init_libnvinfer_plugins(trt_logger, namespace="")

    builder = trt.Builder(trt_logger)
    config = builder.create_builder_config()

    with open(onnx_file, "rb") as fp:
        onnx_model = onnx.load(fp)

    model_props = {}

    model_props["model_version"] = onnx_model.model_version
    model_props["producer_name"] = onnx_model.producer_name
    model_props["producer_version"] = onnx_model.producer_version
    model_props["doc_string"] = onnx_model.doc_string

    log.info(f'Model Version: {model_props["model_version"]}')
    log.info(f'Producer Name: {model_props["producer_name"]}')
    log.info(f'Producer Version: {model_props["producer_version"]}')
    log.info(f'Doc String: {model_props["doc_string"]}')

    for t in onnx_model.metadata_props:
        log.info(f"{t.key}: {t.value}")
        model_props[t.key] = t.value

    props_file = trt_file[:-4] + "_props.json"
    with open(props_file, "w") as f:
        log.info(f"Serializing props to file: {props_file}")
        json.dump(model_props, f)

    network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)

    onnx_path = os.path.realpath(onnx_file)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            log.error(f"Failed to load ONNX file: {onnx_path}")
            for error in range(parser.num_errors):
                log.error(parser.get_error(error))
            raise

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    log.info("Network Description")
    for input in inputs:
        log.info(f"Input '{input.name}' with shape {input.shape} and dtype {input.dtype}")
    for output in outputs:
        log.info(f"Output '{output.name}' with shape {output.shape} and dtype {output.dtype}")

    trt_file = os.path.realpath(trt_file)
    engine_dir = os.path.dirname(trt_file)
    os.makedirs(engine_dir, exist_ok=True)

    log.info(f"Building {precision} Engine in {trt_file}")

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            log.warning("FP16 is not supported natively on this platform/device")
        else:
            config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        # if not builder.platform_has_fast_int8:
        raise Exception("INT8 is not supported natively on this platform/device")

    # builder.max_batch_size = 1 # int(model_props.get("max_num_patches_frame", 2))

    # if network.num_inputs != 1:
    #     raise Exception(f"Invalid num of inputs: {network.num_inputs}")

    if model_props["dataset_type"] != "mask_map":
        optmized_shapes = [1, 1, 1]
        if model_props["dataset_type"] == "object_location":
            optmized_shapes = [1, int(model_props.get("max_num_patches_frame", 2)) // 2, int(model_props.get("max_num_patches_frame", 2))]

        for input in inputs:
            input_shape = list(input.shape)
            profile = builder.create_optimization_profile()
            profile.set_shape(
                input.name,
                ([optmized_shapes[0]] + input_shape[1:]),
                ([optmized_shapes[1]] + input_shape[1:]),
                ([optmized_shapes[2]] + input_shape[1:])
            )
            log.info(f"Profile for Input '{input.name}' shape {input.shape}")

            config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    with open(trt_file, "wb") as f:
        log.info(f"Serializing engine to file: {trt_file}")
        f.write(serialized_engine)

    if model_props["dataset_type"] == "object_location":
        trt_nms_file = trt_file[:-4] + "_nms.trt"

        if "detection_confidence" in component_options:
            model_props["confidence_threshold"] = float(component_options["detection_confidence"])
        elif "confidence_threshold" in component_options:
            model_props["confidence_threshold"] = float(component_options["confidence_threshold"])
        else:
            model_props["confidence_threshold"] = float(model_props["confidence_threshold"])

        if "nms_iou_thresh" in component_options:
            model_props["nms_iou_threshold"] = float(component_options["nms_iou_thresh"])
        elif "nms_iou_threshold" in component_options:
            model_props["nms_iou_threshold"] = float(component_options["nms_iou_threshold"])
        else:
            model_props["nms_iou_threshold"] = float(model_props["nms_iou_threshold"])

        if "max_boxes" in component_options:
            model_props["max_boxes"] = int(component_options["max_boxes"])
        else:
            model_props["max_boxes"] = int(model_props["max_boxes"])

        log.info(f'Creating NMS Engine in {trt_nms_file} - confidence_threshold: {model_props["confidence_threshold"]} - nms_iou_threshold: {model_props["nms_iou_threshold"]} - max_boxes: {model_props["max_boxes"]}')
        create_NMS_engine(
            model_props=model_props,
            engine_path=trt_nms_file,
            output_boxes=network.get_output(0).shape[1],
            output_classes=network.get_output(1).shape[2],
            precision=precision
        )
# ---------------------------------------------------------------------------------------------------------------------------------


def create_tile_plugin_onnx(onnx_file_path):
    opset_version = 15

    patch_size = 160
    input_shape = (1, gs.Tensor.DYNAMIC, gs.Tensor.DYNAMIC, gs.Tensor.DYNAMIC)
    output_channels = 1
    max_image_side = 1600
    max_batch_size = 32
    tiles_shape = (gs.Tensor.DYNAMIC, patch_size, patch_size, output_channels)

    input_image = gs.Variable(name="input", dtype=np.float32, shape=input_shape)
    output_tiles = gs.Variable(name="output_tiles", dtype=np.float32, shape=tiles_shape)

    output_classes = 1
    max_boxes = 30
    confidence_threshold = 0.3
    nms_iou_threshold = 0.4

    exported_model = gs.import_onnx(onnx.load("/opt/eyeflow/data/models/66215adf2c18d39eeadb7f19.onnx"))

    nodes = [
        gs.Node(name="Tile-Input-Image",
                op="TileImage",
                inputs=[input_image],
                outputs=[output_tiles],
                attrs={
                    "patch_size": patch_size,
                    "output_channels": output_channels,
                    "max_image_side": max_image_side,
                    "max_batch_size": max_batch_size
                }
        )
    ]

    # print(exported_model.inputs)
    exported_model.nodes[0].inputs = nodes[0].outputs
    exported_model.nodes[1].inputs[0] = nodes[0].outputs[0]

    nodes.extend(exported_model.nodes)

    # boxes_input_shape = (gs.Tensor.DYNAMIC, gs.Tensor.DYNAMIC, 4)
    # classification_input_shape = (gs.Tensor.DYNAMIC, gs.Tensor.DYNAMIC, gs.Tensor.DYNAMIC)
    input_size_shape = [4]
    boxes_output_shape = (1, gs.Tensor.DYNAMIC, 4)
    classification_output_shape = (1, gs.Tensor.DYNAMIC, output_classes + 1)

    # boxes_input = gs.Variable(name="input_boxes", dtype=np.float32, shape=boxes_input_shape)
    # classification_input = gs.Variable(name="input_classification", dtype=np.float32, shape=classification_input_shape)
    image_size_input = gs.Variable(name="input_image_size", dtype=np.int32, shape=input_size_shape)
    output_boxes = gs.Variable(name="output_boxes", dtype=np.float32, shape=boxes_output_shape)
    output_classification = gs.Variable(name="output_classification", dtype=np.float32, shape=classification_output_shape)

    adjbox_inputs = [image_size_input, image_size_input, image_size_input]
    adjbox_outputs = [output_boxes, output_classification]
    # adjbox_inputs = [image_size_input, image_size_input]
    # adjbox_outputs = [output_boxes]

    # output_classification = None

    for node in nodes:
        if node.outputs[0].name == "decode_boxes":
            # This is the node that outputs the boxes we want to use for NMS
            adjbox_inputs[1] = node.outputs[0]
        elif node.outputs[0].name == "decode_boxes_1":
            # This is the node that outputs the classification scores we want to use for NMS
            adjbox_inputs[2] = node.outputs[0]
            # output_classification = node.outputs[0]

    adjbox_node = gs.Node(name="Adjust-Tiled-Boxes",
                op="AdjustTiledBoxes",
                inputs=adjbox_inputs,
                outputs=adjbox_outputs,
                attrs={
                    "patch_size": patch_size,
                    "max_image_side": max_image_side
                }
        )

    nodes.append(adjbox_node)

    # NMS Outputs.
    nms_output_num_detections = gs.Variable(name="num_detections", dtype=np.int32, shape=[-1, 1])
    nms_output_boxes = gs.Variable(name="detection_boxes", dtype=np.float32, shape=[-1, max_boxes, 4])
    nms_output_scores = gs.Variable(name="detection_scores", dtype=np.float32, shape=[-1, max_boxes])
    nms_output_classes = gs.Variable(name="detection_classes", dtype=np.int32, shape=[-1, max_boxes])

    # nms_inputs = [nms_input_boxes, nms_input_classification]
    nms_outputs = [nms_output_num_detections, nms_output_boxes, nms_output_scores, nms_output_classes]

        # print(f"Node: {node.name}, Inputs: {[i.name for i in node.inputs]}, Outputs: {[o.name for o in node.outputs]}")
        # print(f"Node: {node.name}, Outputs: {[o.name for o in node.outputs]}")
        # print(f"Node: {node.name}, Op: {node.op}, Inputs: {[i.name for i in node.inputs]}, Outputs: {[o.name for o in node.outputs]}")

    # Plugin.
    nms_node = gs.Node(
        op="EfficientNMS_TRT",
        name="nms/non_maximum_suppression",
        inputs=[output_boxes, output_classification],
        outputs=nms_outputs,
        attrs={
            'plugin_version': "1",
            'background_class': -1,
            'max_output_boxes': max_boxes,
            'score_threshold': confidence_threshold,
            'iou_threshold': nms_iou_threshold,
            'score_activation': False,
            'box_coding': 0,
            "class_agnostic": True
        }
    )

    nodes.append(nms_node)

    graph = gs.Graph(nodes=nodes,
                     inputs=[input_image, image_size_input],
                    #  outputs=[output_tiles, output_boxes, output_classification] + nms_outputs,
                     outputs=nms_outputs,
                     opset=opset_version)

    model = gs.export_onnx(graph)
    onnx.save(model, onnx_file_path)

    log.info(f"ONNX file created and saved: {onnx_file_path}")
# ---------------------------------------------------------------------------------------------------------------------------------


def load_plugin_lib(plugin_lib_file_path):

    if os.path.isfile(plugin_lib_file_path):
        try:
            # Python specifies that winmode is 0 by default, but some implementations
            # incorrectly default to None instead. See:
            # https://docs.python.org/3.8/library/ctypes.html
            # https://github.com/python/cpython/blob/3.10/Lib/ctypes/__init__.py#L343
            ctypes.CDLL(plugin_lib_file_path, winmode=0)
        except TypeError:
            # winmode only introduced in python 3.8
            ctypes.CDLL(plugin_lib_file_path)
        return

    raise IOError(f"Failed to load plugin library: {plugin_lib_file_path}")
# ---------------------------------------------------------------------------------------------------------------------------------


def convert_model_trt(onnx_file, plugin_lib_tile_input, plugin_lib_adjust_boxes, trt_file, precision):
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    onnx_file_date = os.path.getmtime(onnx_file)
    if os.path.isfile(trt_file):
        if onnx_file_date < os.path.getmtime(trt_file):
            return

    log.info(f"Converting Model {onnx_file} to TensorRT - Version: {trt.__version__}")
    trt_logger = trt.Logger(min_severity=trt.Logger.INFO)

    trt.init_libnvinfer_plugins(trt_logger, namespace="")
    load_plugin_lib(plugin_lib_tile_input)
    load_plugin_lib(plugin_lib_adjust_boxes)

    builder = trt.Builder(trt_logger)
    config = builder.create_builder_config()

    with open(onnx_file, "rb") as fp:
        onnx_model = onnx.load(fp)

    model_props = {}

    model_props["model_version"] = onnx_model.model_version
    model_props["producer_name"] = onnx_model.producer_name
    model_props["producer_version"] = onnx_model.producer_version
    model_props["doc_string"] = onnx_model.doc_string

    log.info(f'Model Version: {model_props["model_version"]}')
    log.info(f'Producer Name: {model_props["producer_name"]}')
    log.info(f'Producer Version: {model_props["producer_version"]}')
    log.info(f'Doc String: {model_props["doc_string"]}')

    for t in onnx_model.metadata_props:
        log.info(f"{t.key}: {t.value}")
        model_props[t.key] = t.value

    network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)

    log.info("parsing model")
    onnx_path = os.path.realpath(onnx_file)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            log.error(f"Failed to load ONNX file: {onnx_path}")
            for error in range(parser.num_errors):
                log.error(parser.get_error(error))
            raise

    log.info("model parsed")
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    log.info("Network Description")
    for input in inputs:
        log.info(f"Input '{input.name}' with shape {input.shape} and dtype {input.dtype}")
    for output in outputs:
        log.info(f"Output '{output.name}' with shape {output.shape} and dtype {output.dtype}")

    trt_file = os.path.realpath(trt_file)
    engine_dir = os.path.dirname(trt_file)
    os.makedirs(engine_dir, exist_ok=True)

    log.info(f"Building {precision} Engine in {trt_file}")

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            log.warning("FP16 is not supported natively on this platform/device")
        else:
            config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        # if not builder.platform_has_fast_int8:
        raise Exception("INT8 is not supported natively on this platform/device")

    # builder.max_batch_size = 1 # int(model_props.get("max_num_patches_frame", 2))

    # if network.num_inputs != 1:
    #     raise Exception(f"Invalid num of inputs: {network.num_inputs}")

    profile = builder.create_optimization_profile()
    for input in inputs:
        if input.name == "input":
            profile.set_shape(
                input.name,
                ([1, 160, 160, 1]),
                ([1, 1280, 1920, 1]),
                ([1, 2000, 2000, 3])
            )
            log.info(f"Profile for Input '{input.name}' shape {input.shape}")
        elif input.name == "input_image_size":
            profile.set_shape(
                input.name,
                ([4]),
                ([4]),
                ([4])
            )
            log.info(f"Profile for Input '{input.name}' shape {input.shape}")

    config.add_optimization_profile(profile)

    log.info(f"Serializing engine to file: {trt_file}")
    serialized_engine = builder.build_serialized_network(network, config)
    with open(trt_file, "wb") as f:
        f.write(serialized_engine)
# ---------------------------------------------------------------------------------------------------------------------------------






if __name__ == '__main__':
    onnx_file = "./build/roi_location.onnx"
    convert_model_trt_old("/opt/eyeflow/data/models/66215adf2c18d39eeadb7f19.onnx", "/home/tensorrt-test/build/roi_location_old.trt", {"detection_confidence": 0.5, "nms_iou_thresh": 0.5, "max_boxes": 20}, precision="fp32")

    # create_tile_plugin_onnx(onnx_file)
    # convert_model_trt(onnx_file,
    #                 "./build/libtensorrt_plugin-tile_image.so",
    #                 "./build/libtensorrt_plugin-adjust_tiled_boxes.so",
    #                 "./build/roi_location.trt",
    #                 precision="fp32"
    #                 )

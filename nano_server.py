import socket
import struct
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

HOST = "0.0.0.0"
PORT = 5000

ENGINE_PATH = "models/engine/best.engine"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.35
NMS_THRESHOLD = 0.45
JPEG_QUALITY = 80

# CHANGE THESE ONLY IF YOUR MODEL'S CLASS NAMES DIFFER
CLASS_NAMES = [
    "dump_truck",
    "hd_truck",
    "mining_truck",
    "excavator"
]

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def recv_exact(sock, size):
    data = bytearray()

    while len(data) < size:
        chunk = sock.recv(size - len(data))

        if not chunk:
            raise ConnectionError("Connection closed")

        data.extend(chunk)

    return bytes(data)


def load_engine(path):
    with open(path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(f.read())


engine = load_engine(ENGINE_PATH)
context = engine.create_execution_context()

input_idx = None
output_idx = []

for i in range(engine.num_bindings):
    if engine.binding_is_input(i):
        input_idx = i
    else:
        output_idx.append(i)

input_shape = engine.get_binding_shape(input_idx)
output_shape = engine.get_binding_shape(output_idx[0])

print("TensorRT engine loaded")
print("Input:", input_shape)
print("Output:", output_shape)


# Allocate GPU buffers
input_size = trt.volume(input_shape)
input_dtype = trt.nptype(engine.get_binding_dtype(input_idx))

host_input = cuda.pagelocked_empty(input_size, input_dtype)
device_input = cuda.mem_alloc(host_input.nbytes)

host_outputs = []
device_outputs = []

for idx in output_idx:
    shape = engine.get_binding_shape(idx)
    size = trt.volume(shape)
    dtype = trt.nptype(engine.get_binding_dtype(idx))

    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)

    host_outputs.append(host_mem)
    device_outputs.append(device_mem)

stream = cuda.Stream()


def letterbox(image):
    h, w = image.shape[:2]

    scale = min(INPUT_SIZE / w, INPUT_SIZE / h)

    nw = int(w * scale)
    nh = int(h * scale)

    resized = cv2.resize(image, (nw, nh))

    canvas = np.full(
        (INPUT_SIZE, INPUT_SIZE, 3),
        114,
        dtype=np.uint8
    )

    dx = (INPUT_SIZE - nw) // 2
    dy = (INPUT_SIZE - nh) // 2

    canvas[dy:dy + nh, dx:dx + nw] = resized

    return canvas, scale, dx, dy


def inference(frame):

    original_h, original_w = frame.shape[:2]

    image, scale, dx, dy = letterbox(frame)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    np.copyto(host_input, image.ravel())

    bindings = [None] * engine.num_bindings
    bindings[input_idx] = int(device_input)

    for i, idx in enumerate(output_idx):
        bindings[idx] = int(device_outputs[i])

    cuda.memcpy_htod_async(
        device_input,
        host_input,
        stream
    )

    context.execute_async_v2(
        bindings=bindings,
        stream_handle=stream.handle
    )

    for i in range(len(output_idx)):
        cuda.memcpy_dtoh_async(
            host_outputs[i],
            device_outputs[i],
            stream
        )

    stream.synchronize()

    output = host_outputs[0]

    output = output.reshape(output_shape)

    # YOLO output is normally:
    # [1, 4 + number_of_classes, number_of_predictions]

    if output.shape[1] < output.shape[2]:
        output = output[0].transpose(1, 0)
    else:
        output = output[0]

    boxes = []
    scores = []
    class_ids = []

    for row in output:

        cx, cy, w, h = row[:4]

        class_scores = row[4:]

        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        if confidence < CONF_THRESHOLD:
            continue

        x1 = (cx - w / 2 - dx) / scale
        y1 = (cy - h / 2 - dy) / scale
        x2 = (cx + w / 2 - dx) / scale
        y2 = (cy + h / 2 - dy) / scale

        x1 = max(0, min(original_w - 1, int(x1)))
        y1 = max(0, min(original_h - 1, int(y1)))
        x2 = max(0, min(original_w - 1, int(x2)))
        y2 = max(0, min(original_h - 1, int(y2)))

        boxes.append([
            x1,
            y1,
            x2 - x1,
            y2 - y1
        ])

        scores.append(confidence)
        class_ids.append(class_id)

    if not boxes:
        return frame

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        CONF_THRESHOLD,
        NMS_THRESHOLD
    )

    for i in indices:

        i = int(i)

        x, y, w, h = boxes[i]

        class_id = class_ids[i]
        confidence = scores[i]

        if class_id < len(CLASS_NAMES):
            name = CLASS_NAMES[class_id]
        else:
            name = "vehicle"

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        label = "{} {:.2f}".format(
            name,
            confidence
        )

        cv2.putText(
            frame,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return frame


def send_frame(sock, frame):

    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )

    if not ok:
        return

    data = encoded.tobytes()

    sock.sendall(
        struct.pack("!I", len(data))
    )

    sock.sendall(data)


def receive_frame(sock):

    header = recv_exact(sock, 4)

    size = struct.unpack(
        "!I",
        header
    )[0]

    data = recv_exact(sock, size)

    array = np.frombuffer(
        data,
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )

    return frame


server = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM
)

server.setsockopt(
    socket.SOL_SOCKET,
    socket.SO_REUSEADDR,
    1
)

server.bind((HOST, PORT))
server.listen(1)

print("")
print("===================================")
print(" MineGuard Nano AI Server")
print(" Listening on port 5000")
print(" Waiting for Raspberry Pi...")
print("===================================")
print("")

while True:

    client, address = server.accept()

    print("Raspberry Pi connected:", address)

    try:

        while True:

            frame = receive_frame(client)

            if frame is None:
                break

            annotated = inference(frame)

            send_frame(
                client,
                annotated
            )

    except Exception as e:

        print("Connection error:", e)

    finally:

        client.close()

        print("Raspberry Pi disconnected")

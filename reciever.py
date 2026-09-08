import socket
import struct
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import threading
import time


# ============================================================
# MINEGUARD - JETSON NANO AI SERVER
#
# Pipeline:
#
# Raspberry Pi
#      ↓
# TCP receiver thread
#      ↓
# latest JPEG only
#      ↓
# inference thread
#      ↓
# TensorRT GPU
#      ↓
# annotation
#      ↓
# JPEG
#      ↓
# TCP
#      ↓
# Raspberry Pi
#
# IMPORTANT:
# This implementation deliberately drops stale frames.
# It is optimized for LOW LATENCY rather than processing
# every single camera frame.
# ============================================================


# ============================================================
# NETWORK
# ============================================================

HOST = "0.0.0.0"
PORT = 5000

TCP_NODELAY = True

MAX_JPEG_SIZE = 10 * 1024 * 1024


# ============================================================
# MODEL
# ============================================================

ENGINE_PATH = "models/engine/best.engine"

INPUT_SIZE = 640

CONF_THRESHOLD = 0.35
NMS_THRESHOLD = 0.45


# ============================================================
# OUTPUT JPEG
# ============================================================

JPEG_QUALITY = 70


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "dump_truck",
    "hd_truck",
    "mining_truck",
    "excavator"
]


# ============================================================
# TENSORRT
# ============================================================

TRT_LOGGER = trt.Logger(
    trt.Logger.WARNING
)


# ============================================================
# GLOBAL STATE
# ============================================================

running = True

frame_lock = threading.Lock()
frame_condition = threading.Condition(
    frame_lock
)

latest_jpeg = None
latest_frame_id = 0


# ============================================================
# SOCKET FUNCTIONS
# ============================================================

def recv_exact(sock, size):

    """
    Efficient exact-size TCP receive.
    """

    data = bytearray()

    while len(data) < size:

        chunk = sock.recv(
            size - len(data)
        )

        if not chunk:

            raise ConnectionError(
                "Raspberry Pi disconnected"
            )

        data.extend(chunk)

    return bytes(data)


def receive_jpeg(sock):

    """
    Read:
        4-byte frame size
        JPEG payload
    """

    header = recv_exact(
        sock,
        4
    )

    size = struct.unpack(
        "!I",
        header
    )[0]

    if size <= 0 or size > MAX_JPEG_SIZE:

        raise ValueError(
            "Invalid JPEG size: {}".format(
                size
            )
        )

    return recv_exact(
        sock,
        size
    )


def send_jpeg(sock, jpeg_bytes):

    header = struct.pack(
        "!I",
        len(jpeg_bytes)
    )

    sock.sendall(header)
    sock.sendall(jpeg_bytes)


# ============================================================
# LOAD TENSORRT ENGINE
# ============================================================

def load_engine(path):

    print("")
    print("Loading TensorRT engine:")
    print(path)

    with open(
        path,
        "rb"
    ) as f:

        runtime = trt.Runtime(
            TRT_LOGGER
        )

        engine = (
            runtime.deserialize_cuda_engine(
                f.read()
            )
        )

    if engine is None:

        raise RuntimeError(
            "Could not load TensorRT engine"
        )

    return engine


engine = load_engine(
    ENGINE_PATH
)

context = engine.create_execution_context()


# ============================================================
# FIND INPUT / OUTPUT BINDINGS
# ============================================================

input_idx = None
output_idx = []


for i in range(
    engine.num_bindings
):

    if engine.binding_is_input(i):

        input_idx = i

    else:

        output_idx.append(i)


if input_idx is None:

    raise RuntimeError(
        "No TensorRT input binding found"
    )


if len(output_idx) == 0:

    raise RuntimeError(
        "No TensorRT output binding found"
    )


input_shape = engine.get_binding_shape(
    input_idx
)

input_dtype = trt.nptype(
    engine.get_binding_dtype(
        input_idx
    )
)


print("")
print("==========================================")
print("TensorRT engine loaded")
print("==========================================")
print(
    "Input shape :",
    tuple(input_shape)
)

print(
    "Input dtype :",
    input_dtype
)

for idx in output_idx:

    print(
        "Output shape:",
        tuple(
            engine.get_binding_shape(idx)
        )
    )


# ============================================================
# BUFFER ALLOCATION
# ============================================================

input_size = trt.volume(
    input_shape
)

host_input = cuda.pagelocked_empty(
    input_size,
    input_dtype
)

device_input = cuda.mem_alloc(
    host_input.nbytes
)


host_outputs = []
device_outputs = []


for idx in output_idx:

    shape = engine.get_binding_shape(
        idx
    )

    size = trt.volume(
        shape
    )

    dtype = trt.nptype(
        engine.get_binding_dtype(idx)
    )

    host_mem = cuda.pagelocked_empty(
        size,
        dtype
    )

    device_mem = cuda.mem_alloc(
        host_mem.nbytes
    )

    host_outputs.append(
        host_mem
    )

    device_outputs.append(
        device_mem
    )


# One CUDA stream for asynchronous
# H2D -> inference -> D2H.
stream = cuda.Stream()


# Build bindings once.
bindings = [None] * engine.num_bindings

bindings[input_idx] = int(
    device_input
)

for i, idx in enumerate(output_idx):

    bindings[idx] = int(
        device_outputs[i]
    )


# ============================================================
# LETTERBOX
# ============================================================

def letterbox(image):

    h, w = image.shape[:2]

    scale = min(
        INPUT_SIZE / float(w),
        INPUT_SIZE / float(h)
    )

    nw = int(
        w * scale
    )

    nh = int(
        h * scale
    )

    resized = cv2.resize(
        image,
        (nw, nh),
        interpolation=cv2.INTER_LINEAR
    )

    canvas = np.full(
        (
            INPUT_SIZE,
            INPUT_SIZE,
            3
        ),
        114,
        dtype=np.uint8
    )

    dx = (
        INPUT_SIZE - nw
    ) // 2

    dy = (
        INPUT_SIZE - nh
    ) // 2

    canvas[
        dy:dy + nh,
        dx:dx + nw
    ] = resized

    return (
        canvas,
        scale,
        dx,
        dy
    )


# ============================================================
# INFERENCE
# ============================================================

def inference(frame):

    original_h, original_w = (
        frame.shape[:2]
    )


    # --------------------------------------------------------
    # LETTERBOX
    # --------------------------------------------------------

    image, scale, dx, dy = (
        letterbox(frame)
    )


    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # UINT8 -> FLOAT
    #
    # TensorRT input is expected to match the engine dtype.
    # --------------------------------------------------------

    image = image.astype(
        input_dtype
    )


    if input_dtype == np.float32:

        image *= (
            1.0 / 255.0
        )

    elif input_dtype == np.float16:

        image *= (
            np.float16(
                1.0 / 255.0
            )
        )

    else:

        # Most YOLO TensorRT engines use
        # floating point input.
        image = image / 255.0


    # --------------------------------------------------------
    # HWC -> CHW
    # --------------------------------------------------------

    image = np.transpose(
        image,
        (2, 0, 1)
    )


    # --------------------------------------------------------
    # COPY INTO PAGE-LOCKED INPUT
    # --------------------------------------------------------

    np.copyto(
        host_input,
        image.ravel()
    )


    # --------------------------------------------------------
    # GPU INPUT COPY
    # --------------------------------------------------------

    cuda.memcpy_htod_async(
        device_input,
        host_input,
        stream
    )


    # --------------------------------------------------------
    # TENSORRT
    # --------------------------------------------------------

    ok = context.execute_async_v2(
        bindings=bindings,
        stream_handle=stream.handle
    )

    if not ok:

        raise RuntimeError(
            "TensorRT inference failed"
        )


    # --------------------------------------------------------
    # GPU OUTPUT COPY
    # --------------------------------------------------------

    for i in range(
        len(output_idx)
    ):

        cuda.memcpy_dtoh_async(
            host_outputs[i],
            device_outputs[i],
            stream
        )


    # --------------------------------------------------------
    # WAIT FOR CUDA
    # --------------------------------------------------------

    stream.synchronize()


    # --------------------------------------------------------
    # YOLO OUTPUT
    # --------------------------------------------------------

    output = host_outputs[0]

    output = output.reshape(
        tuple(
            engine.get_binding_shape(
                output_idx[0]
            )
        )
    )


    # --------------------------------------------------------
    # HANDLE STANDARD YOLO OUTPUT
    #
    # Typical shape:
    #
    # [1, 4 + classes, predictions]
    #
    # Convert to:
    #
    # [predictions, 4 + classes]
    # --------------------------------------------------------

    if (
        output.ndim == 3
        and output.shape[0] == 1
    ):

        output = output[0]


    if (
        output.ndim == 2
        and output.shape[0] < output.shape[1]
    ):

        output = output.T


    # --------------------------------------------------------
    # VECTORIZED CONFIDENCE CALCULATION
    #
    # Avoid Python loops over every prediction.
    # --------------------------------------------------------

    if output.shape[1] < 5:

        return frame


    boxes_xywh = output[:, :4]

    class_scores = output[:, 4:]


    # Best class per prediction.
    class_ids = np.argmax(
        class_scores,
        axis=1
    )


    confidences = class_scores[
        np.arange(
            class_scores.shape[0]
        ),
        class_ids
    ]


    # Filter BEFORE Python/OpenCV NMS.
    mask = (
        confidences
        >= CONF_THRESHOLD
    )


    if not np.any(mask):

        return frame


    boxes_xywh = boxes_xywh[mask]

    confidences = confidences[mask]

    class_ids = class_ids[mask]


    # --------------------------------------------------------
    # CONVERT YOLO BOXES TO IMAGE COORDINATES
    # --------------------------------------------------------

    cx = boxes_xywh[:, 0]
    cy = boxes_xywh[:, 1]

    bw = boxes_xywh[:, 2]
    bh = boxes_xywh[:, 3]


    x1 = (
        (cx - bw / 2 - dx)
        / scale
    )

    y1 = (
        (cy - bh / 2 - dy)
        / scale
    )

    x2 = (
        (cx + bw / 2 - dx)
        / scale
    )

    y2 = (
        (cy + bh / 2 - dy)
        / scale
    )


    x1 = np.clip(
        x1,
        0,
        original_w - 1
    ).astype(np.int32)

    y1 = np.clip(
        y1,
        0,
        original_h - 1
    ).astype(np.int32)

    x2 = np.clip(
        x2,
        0,
        original_w - 1
    ).astype(np.int32)

    y2 = np.clip(
        y2,
        0,
        original_h - 1
    ).astype(np.int32)


    widths = (
        x2 - x1
    )

    heights = (
        y2 - y1
    )


    # Remove invalid boxes.
    valid = (
        (widths > 0)
        &
        (heights > 0)
    )


    if not np.any(valid):

        return frame


    x1 = x1[valid]
    y1 = y1[valid]
    widths = widths[valid]
    heights = heights[valid]

    confidences = confidences[
        valid
    ]

    class_ids = class_ids[
        valid
    ]


    # --------------------------------------------------------
    # OPEN CV NMS
    #
    # Only a small number of confidence-filtered
    # boxes reach this point.
    # --------------------------------------------------------

    boxes = []

    scores = []

    for i in range(
        len(x1)
    ):

        boxes.append(
            [
                int(x1[i]),
                int(y1[i]),
                int(widths[i]),
                int(heights[i])
            ]
        )

        scores.append(
            float(confidences[i])
        )


    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        CONF_THRESHOLD,
        NMS_THRESHOLD
    )


    if indices is None:

        return frame


    if len(indices) == 0:

        return frame


    # --------------------------------------------------------
    # DRAW
    # --------------------------------------------------------

    for idx in indices:

        # OpenCV can return:
        # [i]
        # or
        # i
        #
        # Normalize it.
        if isinstance(
            idx,
            (list, tuple, np.ndarray)
        ):

            i = int(
                np.asarray(idx).flatten()[0]
            )

        else:

            i = int(idx)


        x = boxes[i][0]
        y = boxes[i][1]

        w = boxes[i][2]
        h = boxes[i][3]

        class_id = int(
            class_ids[i]
        )

        confidence = float(
            scores[i]
        )


        if (
            0 <= class_id
            < len(CLASS_NAMES)
        ):

            name = CLASS_NAMES[
                class_id
            ]

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
            (
                x,
                max(
                    20,
                    y - 8
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


    return frame


# ============================================================
# RECEIVE THREAD
# ============================================================

def receive_worker(client):

    global running
    global latest_jpeg
    global latest_frame_id

    print("[RX] Started")

    while running:

        try:

            # Receive JPEG bytes.
            jpeg_bytes = receive_jpeg(
                client
            )

            # ------------------------------------------------
            # CRITICAL:
            #
            # We DO NOT decode the JPEG here.
            #
            # We only replace the previous JPEG.
            #
            # Therefore, if 5 frames arrive while Nano is
            # processing one frame:
            #
            # old frame -> discarded
            # old frame -> discarded
            # old frame -> discarded
            # old frame -> discarded
            # newest frame -> kept
            # ------------------------------------------------

            with frame_condition:

                latest_jpeg = jpeg_bytes

                latest_frame_id += 1

                frame_condition.notify()

        except Exception as e:

            print(
                "[RX] Connection error:",
                e
            )

            running = False

            with frame_condition:
                frame_condition.notify_all()

            break

    print("[RX] Stopped")


# ============================================================
# INFERENCE / SEND LOOP
# ============================================================

def inference_worker(client):

    global running

    processed_id = 0

    fps_counter = 0
    fps_start = time.perf_counter()


    print("[INFERENCE] Started")


    while running:

        jpeg_bytes = None
        frame_id = 0


        # ----------------------------------------------------
        # WAIT FOR NEWEST FRAME
        # ----------------------------------------------------

        with frame_condition:

            while (
                running
                and latest_frame_id
                == processed_id
            ):

                frame_condition.wait(
                    timeout=0.1
                )


            if not running:

                break


            jpeg_bytes = latest_jpeg

            frame_id = latest_frame_id


        if jpeg_bytes is None:

            continue


        # ----------------------------------------------------
        # JPEG DECODE
        # ----------------------------------------------------

        array = np.frombuffer(
            jpeg_bytes,
            dtype=np.uint8
        )


        frame = cv2.imdecode(
            array,
            cv2.IMREAD_COLOR
        )


        if frame is None:

            processed_id = frame_id

            continue


        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        start = time.perf_counter()

        annotated = inference(
            frame
        )

        inference_ms = (
            time.perf_counter()
            - start
        ) * 1000.0


        # Mark this frame as processed.
        processed_id = frame_id


        # ----------------------------------------------------
        # JPEG ENCODE
        # ----------------------------------------------------

        ok, encoded = cv2.imencode(
            ".jpg",
            annotated,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                JPEG_QUALITY
            ]
        )


        if not ok:

            continue


        jpeg_output = (
            encoded.tobytes()
        )


        # ----------------------------------------------------
        # SEND RESULT
        # ----------------------------------------------------

        try:

            send_jpeg(
                client,
                jpeg_output
            )

        except Exception as e:

            print(
                "[TX] Connection error:",
                e
            )

            running = False

            with frame_condition:
                frame_condition.notify_all()

            break


        # ----------------------------------------------------
        # FPS DISPLAY
        # ----------------------------------------------------

        fps_counter += 1

        elapsed = (
            time.perf_counter()
            - fps_start
        )


        if elapsed >= 2.0:

            fps = (
                fps_counter
                / elapsed
            )

            print(
                "[NANO] FPS: {:.1f} | "
                "TensorRT+processing: {:.1f} ms".format(
                    fps,
                    inference_ms
                )
            )

            fps_counter = 0

            fps_start = time.perf_counter()


    print("[INFERENCE] Stopped")


# ============================================================
# CLIENT HANDLER
# ============================================================

def handle_client(client, address):

    global running
    global latest_jpeg
    global latest_frame_id

    print("")
    print(
        "Raspberry Pi connected:",
        address
    )


    # --------------------------------------------------------
    # SOCKET OPTIONS
    # --------------------------------------------------------

    if TCP_NODELAY:

        client.setsockopt(
            socket.IPPROTO_TCP,
            socket.TCP_NODELAY,
            1
        )


    client.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_SNDBUF,
        1024 * 1024
    )

    client.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_RCVBUF,
        1024 * 1024
    )


    # --------------------------------------------------------
    # RESET FRAME STATE
    # --------------------------------------------------------

    with frame_condition:

        latest_jpeg = None

        latest_frame_id = 0


    # --------------------------------------------------------
    # START RECEIVER
    # --------------------------------------------------------

    receiver_thread = threading.Thread(
        target=receive_worker,
        args=(client,),
        daemon=True
    )


    receiver_thread.start()


    # --------------------------------------------------------
    # INFERENCE LOOP
    #
    # This stays in the current thread so that only one
    # TensorRT execution context touches the GPU at a time.
    # --------------------------------------------------------

    try:

        inference_worker(
            client
        )

    except KeyboardInterrupt:

        pass

    except Exception as e:

        print(
            "[SERVER] Error:",
            e
        )

    finally:

        running = False

        with frame_condition:
            frame_condition.notify_all()

        try:

            client.shutdown(
                socket.SHUT_RDWR
            )

        except:
            pass

        client.close()

        receiver_thread.join(
            timeout=1.0
        )

        print(
            "Raspberry Pi disconnected:",
            address
        )


# ============================================================
# SERVER
# ============================================================

def main():

    global running

    # --------------------------------------------------------
    # SERVER SOCKET
    # --------------------------------------------------------

    server = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    )


    server.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1
    )


    if TCP_NODELAY:

        server.setsockopt(
            socket.IPPROTO_TCP,
            socket.TCP_NODELAY,
            1
        )


    server.bind(
        (HOST, PORT)
    )


    server.listen(1)


    print("")
    print("==========================================")
    print(" MineGuard Nano AI Server")
    print("==========================================")
    print("Listening on:")
    print(
        "{}:{}".format(
            HOST,
            PORT
        )
    )
    print("")
    print("Waiting for Raspberry Pi...")
    print("")


    try:

        while True:

            client, address = (
                server.accept()
            )


            # ------------------------------------------------
            # One client at a time.
            # ------------------------------------------------

            running = True


            handle_client(
                client,
                address
            )


            print("")
            print(
                "Waiting for Raspberry Pi..."
            )


    except KeyboardInterrupt:

        print("")
        print(
            "Server shutting down..."
        )


    finally:

        running = False

        try:
            server.close()
        except:
            pass


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

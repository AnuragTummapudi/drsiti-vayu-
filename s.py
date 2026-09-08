import socket
import struct
import cv2
import numpy as np
import threading
import time


# ============================================================
# MINEGUARD - RASPBERRY PI SENDER
#
# CAMERA:
#     IR Camera -> Raspberry Pi
#
# NETWORK:
#     Pi -> Jetson Nano
#     Jetson Nano -> Pi
#
# DESIGN:
#     - Camera capture stays in MAIN THREAD
#     - Sending runs in background thread
#     - Receiving runs in background thread
#     - Only NEWEST camera frame is kept
#     - Only NEWEST YOLO result is kept
#     - No growing queues
#     - No waiting for YOLO before capturing next frame
#     - Automatic camera recovery if read() fails
# ============================================================


# ============================================================
# NETWORK SETTINGS
# ============================================================

NANO_IP = "192.168.50.1"
NANO_PORT = 5000

MAX_JPEG_SIZE = 10 * 1024 * 1024

SOCKET_BUFFER_SIZE = 1024 * 1024


# ============================================================
# CAMERA SETTINGS
# ============================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 30


# ============================================================
# JPEG
# ============================================================

# 70 is intentionally used for lower latency.
# If image quality is insufficient, use 75.
JPEG_QUALITY = 70


# ============================================================
# DISPLAY
# ============================================================

WINDOW_NAME = "MineGuard - Jetson YOLO"


# ============================================================
# GLOBAL STATE
# ============================================================

running = True


# ------------------------------------------------------------
# Camera -> TX
# ------------------------------------------------------------

frame_lock = threading.Lock()
frame_condition = threading.Condition(frame_lock)

latest_frame = None
latest_frame_id = 0


# ------------------------------------------------------------
# Nano -> Display
# ------------------------------------------------------------

result_lock = threading.Lock()
result_condition = threading.Condition(result_lock)

latest_result = None
latest_result_id = 0


# ============================================================
# TCP HELPERS
# ============================================================

def recv_exact(sock, size):

    data = bytearray()

    while len(data) < size:

        chunk = sock.recv(
            size - len(data)
        )

        if not chunk:
            raise ConnectionError(
                "Jetson Nano disconnected"
            )

        data.extend(chunk)

    return bytes(data)


def send_jpeg(sock, jpeg_data):

    if len(jpeg_data) <= 0:
        return

    if len(jpeg_data) > MAX_JPEG_SIZE:
        raise ValueError(
            "JPEG too large: {} bytes".format(
                len(jpeg_data)
            )
        )

    header = struct.pack(
        "!I",
        len(jpeg_data)
    )

    sock.sendall(header)
    sock.sendall(jpeg_data)


def receive_jpeg(sock):

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


# ============================================================
# CAMERA OPEN
# ============================================================

def open_camera():

    """
    Open camera using the same OpenCV path that your original
    working sender used.

    First try normal OpenCV backend.
    If that fails, try V4L2 explicitly.
    """

    print("")
    print("[CAMERA] Opening camera...")


    # --------------------------------------------------------
    # Attempt 1: OpenCV automatic backend
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if cap.isOpened():

        print(
            "[CAMERA] Opened using default backend"
        )

    else:

        cap.release()

        print(
            "[CAMERA] Default backend failed"
        )

        # ----------------------------------------------------
        # Attempt 2: V4L2
        # ----------------------------------------------------

        cap = cv2.VideoCapture(
            CAMERA_INDEX,
            cv2.CAP_V4L2
        )

        if not cap.isOpened():

            cap.release()

            print(
                "[CAMERA] ERROR: Could not open camera"
            )

            return None


        print(
            "[CAMERA] Opened using V4L2 backend"
        )


    # --------------------------------------------------------
    # Configure camera
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        CAMERA_FPS
    )


    # --------------------------------------------------------
    # Buffer size
    #
    # This property is backend-dependent. We do NOT rely on it.
    # If supported, setting it to 1 helps reduce stale frames.
    # --------------------------------------------------------

    try:

        cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

    except Exception:

        pass


    # --------------------------------------------------------
    # Print actual camera configuration
    # --------------------------------------------------------

    actual_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    actual_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    actual_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    backend = cap.getBackendName()


    print(
        "[CAMERA] Backend     : {}"
        .format(backend)
    )

    print(
        "[CAMERA] Resolution  : {}x{}"
        .format(
            actual_width,
            actual_height
        )
    )

    print(
        "[CAMERA] Camera FPS  : {:.1f}"
        .format(
            actual_fps
        )
    )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Actually test a frame before declaring the camera ready.
    # --------------------------------------------------------

    for attempt in range(10):

        ret, frame = cap.read()

        if ret and frame is not None:

            print(
                "[CAMERA] First frame received"
            )

            print(
                "[CAMERA] Frame shape: {}"
                .format(
                    frame.shape
                )
            )

            return cap, frame


        print(
            "[CAMERA] Waiting for frame... "
            "attempt {}/10"
            .format(
                attempt + 1
            )
        )

        time.sleep(0.1)


    print(
        "[CAMERA] ERROR: Camera opened but "
        "no frames were received."
    )

    cap.release()

    return None


# ============================================================
# TRANSMIT WORKER
# ============================================================

def send_worker(sock):

    global running

    last_sent_id = 0

    print("[TX] Sender thread started")


    while running:

        # ----------------------------------------------------
        # Wait for NEW camera frame
        # ----------------------------------------------------

        with frame_condition:

            while (
                running
                and latest_frame_id
                == last_sent_id
            ):

                frame_condition.wait(
                    timeout=0.1
                )


            if not running:
                break


            frame = latest_frame

            frame_id = latest_frame_id

            last_sent_id = frame_id


        if frame is None:
            continue


        try:

            # ------------------------------------------------
            # JPEG ENCODE
            # ------------------------------------------------

            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    JPEG_QUALITY
                ]
            )


            if not ok:

                print(
                    "[TX] JPEG encode failed"
                )

                continue


            jpeg_data = encoded.tobytes()


            # ------------------------------------------------
            # SEND
            # ------------------------------------------------

            send_jpeg(
                sock,
                jpeg_data
            )


        except Exception as e:

            print(
                "[TX] ERROR:",
                e
            )

            running = False

            with frame_condition:
                frame_condition.notify_all()

            with result_condition:
                result_condition.notify_all()

            break


    print("[TX] Sender thread stopped")


# ============================================================
# RECEIVE WORKER
# ============================================================

def receive_worker(sock):

    global running
    global latest_result
    global latest_result_id


    print("[RX] Receiver thread started")


    while running:

        try:

            # ------------------------------------------------
            # Receive newest annotated JPEG
            # ------------------------------------------------

            jpeg_data = receive_jpeg(
                sock
            )


            # ------------------------------------------------
            # IMPORTANT:
            #
            # Do NOT decode here.
            #
            # Just replace old result with newest result.
            # ------------------------------------------------

            with result_condition:

                latest_result = jpeg_data

                latest_result_id += 1

                result_condition.notify()


        except Exception as e:

            if running:

                print(
                    "[RX] ERROR:",
                    e
                )


            running = False

            with frame_condition:
                frame_condition.notify_all()

            with result_condition:
                result_condition.notify_all()

            break


    print("[RX] Receiver thread stopped")


# ============================================================
# MAIN
# ============================================================

def main():

    global running
    global latest_frame
    global latest_frame_id


    print("")
    print("==============================================")
    print("       MineGuard Raspberry Pi Client")
    print("==============================================")
    print("")
    print(
        "Jetson Nano: {}:{}"
        .format(
            NANO_IP,
            NANO_PORT
        )
    )
    print(
        "Camera: {}x{} @ {} FPS"
        .format(
            FRAME_WIDTH,
            FRAME_HEIGHT,
            CAMERA_FPS
        )
    )
    print(
        "JPEG quality: {}"
        .format(
            JPEG_QUALITY
        )
    )
    print("")
    print("Architecture:")
    print("Camera -> Pi -> Nano -> Pi -> Display")
    print("Low-latency latest-frame mode")
    print("")


    # ========================================================
    # CAMERA
    # ========================================================

    camera_result = open_camera()

    if camera_result is None:

        print("")
        print(
            "FATAL: Camera could not be started."
        )
        print(
            "Check the camera connection/device."
        )

        return


    cap, first_frame = camera_result


    # Put first frame into latest-frame buffer.
    with frame_condition:

        latest_frame = first_frame

        latest_frame_id = 1

        frame_condition.notify()


    # ========================================================
    # CONNECT TO NANO
    # ========================================================

    print("")
    print(
        "[NETWORK] Connecting to Jetson Nano..."
    )


    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    )


    # Disable Nagle.
    sock.setsockopt(
        socket.IPPROTO_TCP,
        socket.TCP_NODELAY,
        1
    )


    # Larger TCP buffers.
    sock.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_SNDBUF,
        SOCKET_BUFFER_SIZE
    )

    sock.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_RCVBUF,
        SOCKET_BUFFER_SIZE
    )


    try:

        sock.connect(
            (
                NANO_IP,
                NANO_PORT
            )
        )

    except Exception as e:

        print(
            "[NETWORK] Connection failed:",
            e
        )

        cap.release()

        sock.close()

        return


    print(
        "[NETWORK] Connected to Jetson Nano!"
    )


    # ========================================================
    # START NETWORK THREADS
    # ========================================================

    tx_thread = threading.Thread(
        target=send_worker,
        args=(sock,),
        daemon=True
    )


    rx_thread = threading.Thread(
        target=receive_worker,
        args=(sock,),
        daemon=True
    )


    tx_thread.start()
    rx_thread.start()


    print("")
    print("==============================================")
    print("           STREAMING STARTED")
    print("==============================================")
    print("")
    print("Camera capture : MAIN THREAD")
    print("Network TX     : BACKGROUND")
    print("Network RX     : BACKGROUND")
    print("Frame policy   : LATEST FRAME ONLY")
    print("")
    print("Press Q to quit")
    print("")


    # ========================================================
    # MAIN CAMERA + DISPLAY LOOP
    #
    # CRITICAL:
    #
    # cap.read() stays in the main thread.
    #
    # It NEVER waits for Nano.
    # ========================================================

    displayed_result_id = 0

    camera_frames = 0

    display_frames = 0

    fps_start = time.perf_counter()

    last_status = time.perf_counter()

    consecutive_read_failures = 0


    try:

        while running:

            # =================================================
            # 1. CAPTURE
            # =================================================

            ret, frame = cap.read()


            if not ret or frame is None:

                consecutive_read_failures += 1

                print(
                    "[CAMERA] Frame read failed "
                    "({})"
                    .format(
                        consecutive_read_failures
                    )
                )


                # ---------------------------------------------
                # Attempt camera recovery after repeated failure
                # ---------------------------------------------

                if consecutive_read_failures >= 10:

                    print(
                        "[CAMERA] Attempting camera recovery..."
                    )


                    cap.release()

                    time.sleep(0.5)


                    camera_result = open_camera()


                    if camera_result is None:

                        print(
                            "[CAMERA] Recovery failed."
                        )

                        time.sleep(1.0)

                        continue


                    cap, frame = camera_result


                    consecutive_read_failures = 0


                else:

                    continue


            else:

                consecutive_read_failures = 0


            camera_frames += 1


            # =================================================
            # 2. UPDATE LATEST FRAME
            #
            # The TX thread will pick this up.
            #
            # We DO NOT wait for TX.
            # =================================================

            with frame_condition:

                latest_frame = frame

                latest_frame_id += 1

                frame_condition.notify()


            # =================================================
            # 3. GET LATEST YOLO RESULT
            # =================================================

            jpeg_data = None

            current_result_id = 0


            with result_lock:

                if (
                    latest_result is not None
                    and latest_result_id
                    != displayed_result_id
                ):

                    jpeg_data = latest_result

                    current_result_id = (
                        latest_result_id
                    )


            # =================================================
            # 4. DISPLAY LATEST RESULT
            # =================================================

            if jpeg_data is not None:

                array = np.frombuffer(
                    jpeg_data,
                    dtype=np.uint8
                )


                annotated = cv2.imdecode(
                    array,
                    cv2.IMREAD_COLOR
                )


                if annotated is not None:

                    displayed_result_id = (
                        current_result_id
                    )


                    cv2.imshow(
                        WINDOW_NAME,
                        annotated
                    )


                    display_frames += 1


            # =================================================
            # 5. IMPORTANT:
            #
            # Keep OpenCV GUI responsive.
            # =================================================

            key = cv2.waitKey(1) & 0xFF


            if key == ord("q"):

                break


            # =================================================
            # 6. STATUS
            # =================================================

            now = time.perf_counter()


            if (
                now - last_status
                >= 2.0
            ):

                elapsed = (
                    now - fps_start
                )


                if elapsed > 0:

                    camera_fps = (
                        camera_frames
                        / elapsed
                    )


                    display_fps = (
                        display_frames
                        / elapsed
                    )


                else:

                    camera_fps = 0.0
                    display_fps = 0.0


                print(
                    "[PI] Camera: {:.1f} FPS | "
                    "Displayed YOLO: {:.1f} FPS | "
                    "Camera frame ID: {} | "
                    "Result ID: {}"
                    .format(
                        camera_fps,
                        display_fps,
                        latest_frame_id,
                        latest_result_id
                    )
                )


                camera_frames = 0
                display_frames = 0

                fps_start = now
                last_status = now


    except KeyboardInterrupt:

        pass


    except Exception as e:

        print(
            "[MAIN] ERROR:",
            e
        )


    finally:

        print("")
        print(
            "[SYSTEM] Shutting down..."
        )


        running = False


        with frame_condition:
            frame_condition.notify_all()


        with result_condition:
            result_condition.notify_all()


        try:

            sock.shutdown(
                socket.SHUT_RDWR
            )

        except Exception:

            pass


        try:

            sock.close()

        except Exception:

            pass


        try:

            cap.release()

        except Exception:

            pass


        cv2.destroyAllWindows()


        print(
            "[SYSTEM] MineGuard client stopped."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

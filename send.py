import socket
import struct
import cv2
import numpy as np
import threading
import time

# ============================================================
# MINEGUARD - RASPBERRY PI CLIENT
#
# Camera:
#       IR Camera -> Pi
#
# Network:
#       Pi -> Jetson Nano -> Pi
#
# Design:
#       - Camera capture runs independently
#       - Network sending runs independently
#       - Network receiving runs independently
#       - Only the newest frame is kept
#       - Old/stale frames are automatically dropped
#       - No waiting for Nano before capturing the next frame
# ============================================================


# ==============================
# NETWORK SETTINGS
# ==============================

NANO_IP = "192.168.50.2"
NANO_PORT = 5000

SOCKET_TIMEOUT = 5.0

# Disable Nagle's algorithm.
# This is useful for small control/header packets and
# reduces packet coalescing delay.
TCP_NODELAY = True


# ==============================
# CAMERA SETTINGS
# ==============================

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 30


# ==============================
# JPEG SETTINGS
# ==============================

# 70-75 is a good latency/quality compromise.
JPEG_QUALITY = 70


# ==============================
# DISPLAY
# ==============================

WINDOW_NAME = "MineGuard - Jetson YOLO"


# ==============================
# GLOBAL STATE
# ==============================

running = True

frame_lock = threading.Lock()
frame_condition = threading.Condition(frame_lock)

latest_frame = None
latest_frame_id = 0


result_lock = threading.Lock()
result_condition = threading.Condition(result_lock)

latest_result = None
latest_result_id = 0


# ============================================================
# SOCKET HELPERS
# ============================================================

def recv_exact(sock, size):
    """
    Receive exactly 'size' bytes.

    Uses bytearray instead of repeatedly doing:
        data += chunk

    which avoids unnecessary byte-string reallocations.
    """

    data = bytearray()

    while len(data) < size:

        chunk = sock.recv(size - len(data))

        if not chunk:
            raise ConnectionError("Jetson Nano disconnected")

        data.extend(chunk)

    return bytes(data)


def send_jpeg(sock, jpeg_bytes):
    """
    Send:
        4-byte network-order frame size
        JPEG payload
    """

    header = struct.pack("!I", len(jpeg_bytes))

    sock.sendall(header)
    sock.sendall(jpeg_bytes)


def receive_jpeg(sock):
    """
    Receive:
        4-byte size
        JPEG payload

    Returns JPEG bytes.
    """

    header = recv_exact(sock, 4)

    size = struct.unpack("!I", header)[0]

    # Basic protection against corrupted/invalid packets.
    if size <= 0 or size > 10 * 1024 * 1024:
        raise ValueError(
            "Invalid JPEG packet size: {}".format(size)
        )

    return recv_exact(sock, size)


# ============================================================
# CAMERA THREAD
# ============================================================

def camera_worker(cap):

    global running
    global latest_frame
    global latest_frame_id

    print("[CAMERA] Started")

    while running:

        ret, frame = cap.read()

        if not ret:
            continue

        # Only keep the newest frame.
        #
        # We intentionally do NOT create a queue.
        # If Nano is slower than the camera, old frames are discarded.
        with frame_condition:

            latest_frame = frame
            latest_frame_id += 1

            frame_condition.notify()

    print("[CAMERA] Stopped")


# ============================================================
# SEND THREAD
# ============================================================

def send_worker(sock):

    global running

    last_sent_id = 0

    print("[TX] Started")

    while running:

        # Wait for a new camera frame.
        with frame_condition:

            while running and latest_frame_id == last_sent_id:
                frame_condition.wait(timeout=0.1)

            if not running:
                break

            frame = latest_frame
            frame_id = latest_frame_id

            # Important:
            # We copy the reference while holding the lock.
            # The camera will replace latest_frame with a new ndarray,
            # so this frame remains valid for this worker.
            last_sent_id = frame_id

        if frame is None:
            continue

        try:

            # JPEG compression happens outside the lock.
            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    JPEG_QUALITY
                ]
            )

            if not ok:
                continue

            jpeg_bytes = encoded.tobytes()

            send_jpeg(sock, jpeg_bytes)

        except Exception as e:

            print("[TX] Error:", e)

            running = False

            with frame_condition:
                frame_condition.notify_all()

            break

    print("[TX] Stopped")


# ============================================================
# RECEIVE THREAD
# ============================================================

def receive_worker(sock):

    global running
    global latest_result
    global latest_result_id

    print("[RX] Started")

    while running:

        try:

            jpeg_bytes = receive_jpeg(sock)

            # Do NOT immediately decode every frame.
            #
            # Store only the newest JPEG.
            #
            # If Nano somehow produces frames faster than
            # the display can consume them, stale results disappear.
            with result_condition:

                latest_result = jpeg_bytes
                latest_result_id += 1

                result_condition.notify()

        except Exception as e:

            print("[RX] Error:", e)

            running = False

            with result_condition:
                result_condition.notify_all()

            break

    print("[RX] Stopped")


# ============================================================
# MAIN
# ============================================================

def main():

    global running

    # --------------------------------------------------------
    # CONNECT TO NANO
    # --------------------------------------------------------

    print("")
    print("==========================================")
    print(" MineGuard Raspberry Pi Client")
    print("==========================================")
    print("Connecting to Jetson Nano...")
    print("IP   :", NANO_IP)
    print("PORT :", NANO_PORT)

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    )

    # Prevent delayed TCP packet coalescing.
    if TCP_NODELAY:
        sock.setsockopt(
            socket.IPPROTO_TCP,
            socket.TCP_NODELAY,
            1
        )

    # Larger socket buffers are useful for JPEG video.
    sock.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_SNDBUF,
        1024 * 1024
    )

    sock.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_RCVBUF,
        1024 * 1024
    )

    sock.settimeout(SOCKET_TIMEOUT)

    sock.connect(
        (NANO_IP, NANO_PORT)
    )

    # After connect, blocking I/O is preferable.
    # The worker threads handle blocking independently.
    sock.settimeout(None)

    print("Connected!")
    print("")


    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    cap = cv2.VideoCapture(0)

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

    # Reduce internal camera buffering when the backend supports it.
    # This helps prevent OpenCV/V4L2 from handing us very old frames.
    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    if not cap.isOpened():

        print("ERROR: Camera could not be opened")

        sock.close()

        return


    print("Camera started")
    print(
        "Requested resolution: {}x{}".format(
            FRAME_WIDTH,
            FRAME_HEIGHT
        )
    )

    print(
        "Requested FPS: {}".format(
            CAMERA_FPS
        )
    )

    print("")
    print("Streaming:")
    print("Camera -> Pi -> Nano -> Pi -> Display")
    print("")
    print("Press Q to quit")
    print("")


    # --------------------------------------------------------
    # START THREADS
    # --------------------------------------------------------

    camera_thread = threading.Thread(
        target=camera_worker,
        args=(cap,),
        daemon=True
    )

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

    camera_thread.start()
    tx_thread.start()
    rx_thread.start()


    # --------------------------------------------------------
    # DISPLAY LOOP
    #
    # Keep GUI work in the main thread.
    # --------------------------------------------------------

    displayed_result_id = 0

    fps_counter = 0
    fps_start = time.perf_counter()

    display_fps = 0.0

    try:

        while running:

            jpeg_bytes = None
            result_id = 0

            # Wait briefly for a result.
            with result_condition:

                if latest_result_id == displayed_result_id:

                    result_condition.wait(
                        timeout=0.05
                    )

                if latest_result is not None:

                    jpeg_bytes = latest_result
                    result_id = latest_result_id

            if jpeg_bytes is None:
                continue

            # Decode only the newest available result.
            array = np.frombuffer(
                jpeg_bytes,
                dtype=np.uint8
            )

            annotated = cv2.imdecode(
                array,
                cv2.IMREAD_COLOR
            )

            if annotated is None:
                continue

            displayed_result_id = result_id

            cv2.imshow(
                WINDOW_NAME,
                annotated
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break


            # ------------------------------------------------
            # DISPLAY FPS
            # ------------------------------------------------

            fps_counter += 1

            elapsed = (
                time.perf_counter()
                - fps_start
            )

            if elapsed >= 2.0:

                display_fps = (
                    fps_counter / elapsed
                )

                print(
                    "[DISPLAY] {:.1f} FPS".format(
                        display_fps
                    )
                )

                fps_counter = 0
                fps_start = time.perf_counter()


    except KeyboardInterrupt:

        pass

    except Exception as e:

        print(
            "[MAIN] Error:",
            e
        )


    # --------------------------------------------------------
    # SHUTDOWN
    # --------------------------------------------------------

    running = False

    with frame_condition:
        frame_condition.notify_all()

    with result_condition:
        result_condition.notify_all()

    try:
        sock.shutdown(socket.SHUT_RDWR)
    except:
        pass

    sock.close()

    cap.release()

    cv2.destroyAllWindows()

    print("")
    print("MineGuard client stopped.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

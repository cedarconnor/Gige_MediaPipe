import cv2
import numpy as np
import mvsdk
import platform
import mediapipe as mp
from pythonosc import udp_client

# Initialize MediaPipe and OSC client
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
client = udp_client.SimpleUDPClient("127.0.0.1", 8888)


def main_loop():
    # Enumerate cameras
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    if nDev < 1:
        print("No camera was found!")
        return

    DevInfo = DevList[0]
    hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
    cap = mvsdk.CameraGetCapability(hCamera)
    monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

    if monoCamera:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
    else:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

    mvsdk.CameraSetTriggerMode(hCamera, 0)
    mvsdk.CameraSetAeState(hCamera, 0)
    mvsdk.CameraSetExposureTime(hCamera, 20 * 1000)
    mvsdk.CameraPlay(hCamera)

    FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)
    pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

    # Initialize MediaPipe Hands
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.2,
            max_num_hands=8) as hands:

        while (cv2.waitKey(1) & 0xFF) != ord('q'):
            try:
                pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
                mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
                mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

                if platform.system() == "Windows":
                    mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)

                frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
                frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth,
                                       1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))

                # Convert single-channel grayscale to three-channel BGR
                if frame.shape[-1] == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                if results.multi_hand_landmarks:
                    for hand_id, hand_lm in enumerate(results.multi_hand_landmarks):
                        mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

                        for landmark_id, lm in enumerate(hand_lm.landmark):
                            if landmark_id == 9:  # MIDDLE_FINGER_MCP
                                coords = [lm.x, lm.y, lm.z]
                                client.send_message(f"/hand/{hand_id}/{landmark_id}", coords)

                resized_frame = cv2.resize(frame, (1280, 1024), interpolation=cv2.INTER_LINEAR)
                cv2.imshow("Press q to end", resized_frame)

            except mvsdk.CameraException as e:
                if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                    print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))

    mvsdk.CameraUnInit(hCamera)
    mvsdk.CameraAlignFree(pFrameBuffer)


def main():
    try:
        main_loop()
    finally:
        cv2.destroyAllWindows()


main()

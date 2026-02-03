# ==================== IMPORT ====================
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# ==================== LOAD MODEL ====================
model_path = r"D:\Projects\sibi_sign_language_model\Model\action.h5"
model = tf.keras.models.load_model(
    r"D:\Projects\sibi_sign_language_model\Model\action.h5"
)


# ==================== ACTION CLASSES ====================
actions = np.array(['hello', 'thankyou', 'iloveyou'])
colors = [(245,117,16), (117,245,16), (16,117,245)]
threshold = 0.5

# ==================== MEDIAPIPE SETUP ====================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ==================== FUNCTIONS ====================
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

def extract_keypoints(results):
    pose = np.array(
        [[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]
    ).flatten() if results.pose_landmarks else np.zeros(33 * 4)

    face = np.array(
        [[res.x, res.y, res.z] for res in results.face_landmarks.landmark]
    ).flatten() if results.face_landmarks else np.zeros(468 * 3)

    lh = np.array(
        [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
    ).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)

    rh = np.array(
        [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
    ).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

    return np.concatenate([pose, face, lh, rh])

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()

    for num, prob in enumerate(res):
        percentage = int(prob * 100)

        # Bar
        cv2.rectangle(
            output_frame,
            (0, 60 + num * 40),
            (percentage * 2, 90 + num * 40),  # diperlebar biar jelas
            colors[num],
            -1
        )

        # Nama class
        cv2.putText(
            output_frame,
            actions[num],
            (10, 85 + num * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Persentase
        cv2.putText(
            output_frame,
            f"{percentage}%",
            (220, 85 + num * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return output_frame

def get_camera_index(max_test=5):
    for i in range(max_test):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print(f"✅ Camera ditemukan di index {i}")
                return i
    return -1


# ==================== REALTIME DETECTION ====================
sequence = []
sentence = []
predictions = []

cam_index = get_camera_index()
if cam_index == -1:
    print("❌ Tidak ada kamera terdeteksi")
    exit()

cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)


with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(
                np.expand_dims(sequence, axis=0),
                verbose=0
            )[0]

            pred = np.argmax(res)
            predictions.append(pred)

            if np.unique(predictions[-10:])[0] == pred:
                if res[pred] > threshold:
                    if len(sentence) > 0:
                        if actions[pred] != sentence[-1]:
                            sentence.append(actions[pred])
                    else:
                        sentence.append(actions[pred])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            image = prob_viz(res, actions, image, colors)

        cv2.rectangle(image, (0,0), (640,40), (245,117,16), -1)
        cv2.putText(
            image,
            ' '.join(sentence),
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow('SIBI Action Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()



from ultralytics import YOLO
import cv2
import os
import numpy as np
import time

# ----------------- CONFIG -----------------
MODEL_PATH = "yolov10n.pt"
FACES_DIR = "faces"
TRAINER_FILE = "trainer.yml"
LABEL_FILE = "face_labels.txt"
CONF_THRESHOLD = 60
# ------------------------------------------

os.makedirs(FACES_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()

label_map = {}

if os.path.exists(LABEL_FILE):
    with open(LABEL_FILE, "r") as f:
        for line in f:
            name, id = line.strip().split(",")
            label_map[int(id)] = name

if os.path.exists(TRAINER_FILE):
    recognizer.read(TRAINER_FILE)


# ----------------- AUGMENTATION -----------------

def augment_face(face):

    augmented = []

    h, w = face.shape

    low_brightness = cv2.convertScaleAbs(face, alpha=0.7, beta=-30)
    high_brightness = cv2.convertScaleAbs(face, alpha=1.3, beta=30)

    zoom_in = cv2.resize(face[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)], (w,h))

    zoom_out = cv2.resize(face, None, fx=0.8, fy=0.8)
    zoom_out = cv2.copyMakeBorder(
        zoom_out,
        (h - zoom_out.shape[0])//2,
        (h - zoom_out.shape[0])//2,
        (w - zoom_out.shape[1])//2,
        (w - zoom_out.shape[1])//2,
        cv2.BORDER_CONSTANT
    )

    M_left = np.float32([[1,0,-10],[0,1,0]])
    M_right = np.float32([[1,0,10],[0,1,0]])
    M_up = np.float32([[1,0,0],[0,1,-10]])
    M_down = np.float32([[1,0,0],[0,1,10]])

    move_left = cv2.warpAffine(face, M_left, (w,h))
    move_right = cv2.warpAffine(face, M_right, (w,h))
    move_up = cv2.warpAffine(face, M_up, (w,h))
    move_down = cv2.warpAffine(face, M_down, (w,h))

    M_rot_r = cv2.getRotationMatrix2D((w//2,h//2), 10, 1)
    M_rot_l = cv2.getRotationMatrix2D((w//2,h//2), -10, 1)

    rot_r = cv2.warpAffine(face, M_rot_r, (w,h))
    rot_l = cv2.warpAffine(face, M_rot_l, (w,h))

    flip = cv2.flip(face, 1)

    augmented.extend([
        low_brightness,
        high_brightness,
        zoom_in,
        zoom_out,
        move_left,
        move_right,
        move_up,
        move_down,
        rot_r,
        rot_l,
        flip
    ])

    return augmented


# ----------------- TRAINING -----------------

def train_faces():

    start_time = time.time()   # START TIMER

    faces = []
    labels = []

    label_ids = {}
    current_id = 0

    for person in os.listdir(FACES_DIR):

        path = os.path.join(FACES_DIR, person)

        if person not in label_ids:
            label_ids[person] = current_id
            current_id += 1

        label_id = label_ids[person]

        for img_name in os.listdir(path):

            img_path = os.path.join(path, img_name)

            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if gray is None:
                continue

            faces.append(gray)
            labels.append(label_id)

    if len(faces) == 0:
        return

    recognizer.train(faces, np.array(labels))
    recognizer.save(TRAINER_FILE)

    with open(LABEL_FILE, "w") as f:
        for name, id in label_ids.items():
            f.write(f"{name},{id}\n")

    label_map.clear()

    for name, id in label_ids.items():
        label_map[id] = name

    end_time = time.time()   # END TIMER
    training_time = end_time - start_time

    print("Training complete")
    print(f"Training time: {training_time:.2f} seconds")


train_faces()


# ----------------- CAMERA -----------------

cap = cv2.VideoCapture(0)

current_name = "Unknown"
face_img = None

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    for r in results:

        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):

            if int(cls) == 0:

                x1,y1,x2,y2 = map(int,box)

                x1,y1 = max(0,x1),max(0,y1)
                x2,y2 = min(frame.shape[1],x2),min(frame.shape[0],y2)

                person = frame[y1:y2,x1:x2]

                gray = cv2.cvtColor(person,cv2.COLOR_BGR2GRAY)

                faces = face_detector.detectMultiScale(gray,1.3,5)

                for (fx,fy,fw,fh) in faces:

                    face_img = gray[fy:fy+fh,fx:fx+fw]

                    try:

                        id,conf = recognizer.predict(face_img)

                        if conf < CONF_THRESHOLD and id in label_map:
                            current_name = label_map[id]
                            accuracy = max(0, 100 - conf)
                            confidence_text = f"{current_name}: {accuracy:.1f}%"
                        else:
                            current_name = "Unknown"
                            accuracy = max(0, 100 - conf)
                            confidence_text = f"Unknown: {accuracy:.1f}%"

                    except:
                        current_name = "Unknown"
                        confidence_text = "Unknown: N/A"

                    cv2.rectangle(person,(fx,fy),(fx+fw,fy+fh),(0,255,0),2)

                    cv2.putText(
                        person,
                        confidence_text,
                        (fx,fy-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,255,0),
                        2
                    )

                frame[y1:y2,x1:x2] = person

    cv2.putText(frame,"Y = Ordered | N = New User | ESC = Exit",
                (20,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    if current_name != "Unknown":
        cv2.putText(frame,f"Hi {current_name} Did you order?",
                    (20,70),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

    cv2.imshow("Smart Ordering Camera",frame)

    key = cv2.waitKey(1) & 0xFF


# ----------------- SAVE ORDERED -----------------

    if key == ord("y") and current_name != "Unknown" and face_img is not None:

        save_path = os.path.join(FACES_DIR, current_name)
        os.makedirs(save_path, exist_ok=True)

        face_id = len(os.listdir(save_path))

        # Save original face
        cv2.imwrite(f"{save_path}/{current_name}_{face_id}.jpg", face_img)
        face_id += 1

        # 🔥 Apply augmentation (this is what you want)
        augmented_faces = augment_face(face_img)

        for aug in augmented_faces:
            cv2.imwrite(f"{save_path}/{current_name}_{face_id}.jpg", aug)
            face_id += 1

        train_faces()

        print(f"Added 1 face + {len(augmented_faces)} augmented images")


# ----------------- REGISTER NEW USER -----------------

    elif key == ord("n") and face_img is not None:

        name = input("Enter new user name: ")

        save_path = os.path.join(FACES_DIR,name)

        os.makedirs(save_path,exist_ok=True)

        print("Capturing 5 faces...")

        captured = []

        while len(captured) < 5:

            ret,frame = cap.read()

            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            faces = face_detector.detectMultiScale(gray,1.3,5)

            for (fx,fy,fw,fh) in faces:

                face = gray[fy:fy+fh,fx:fx+fw]

                captured.append(face)

                print("Captured",len(captured),"/5")

                cv2.rectangle(frame,(fx,fy),(fx+fw,fy+fh),(0,255,0),2)

                cv2.imshow("Capture",frame)

                cv2.waitKey(300)

                if len(captured) == 5:
                    break

        face_id = 0

        for face in captured:

            cv2.imwrite(f"{save_path}/{name}_{face_id}.jpg",face)
            face_id += 1

            augmented = augment_face(face)

            for aug in augmented:

                cv2.imwrite(f"{save_path}/{name}_{face_id}.jpg",aug)
                face_id += 1

        train_faces()

        print(f"{name} registered with augmented dataset")


# ----------------- EXIT -----------------

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
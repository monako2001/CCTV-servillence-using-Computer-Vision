import cv2
import numpy as np
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Dropout, Dense, Input, Lambda
import keras.backend as k
from datetime import datetime
import tensorflow as tf
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from mtcnn.mtcnn import MTCNN

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

img_size = (32, 32, 3)

def get_feature_extractor(size = img_size):
    """
    Function to get the feature extractor model for the Siamese network for Dress recognition
    """
    extractor = Sequential()
    extractor.add(Input(shape = size))
    extractor.add(Conv2D(128, (3, 3), activation='relu', padding = 'valid'))
    extractor.add(Conv2D(128, (3, 3), activation='relu', padding = 'valid'))
    extractor.add(MaxPooling2D(pool_size=(2, 2)))
    extractor.add(BatchNormalization())
    extractor.add(Dropout(0.3))

    extractor.add(Conv2D(256, (3, 3), activation='relu'))
    extractor.add(Conv2D(256, (3, 3), activation='relu'))
    extractor.add(MaxPooling2D(pool_size=(2, 2)))
    extractor.add(BatchNormalization())
    extractor.add(Dropout(0.3))



    extractor.add(GlobalAveragePooling2D())
    extractor.add(Dense(512))
    extractor.add(Dense(128))

    return extractor


def load_yolo():
    """
    Loads the YOLO object detector
    """
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers


def detect_objects(img, net, outputLayers):	
    """
    Detecting objects and giving outputs
    """		
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs

def get_box_dimensions(outputs, height, width):
    """
    Getting the boxes and filtering out target classes to be detected(Here Person)
    """
    boxes = []
    confs = []
    classes = [0]
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if class_id in classes:
                if conf > 0.3:
                    center_x = int(detect[0] * width)
                    center_y = int(detect[1] * height)
                    w = int(detect[2] * width)
                    h = int(detect[3] * height)
                    x = int(center_x - w/2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confs.append(float(conf))
                    class_ids.append(class_id)
    return boxes, confs, class_ids



def check_dress(img, target_dress, dress_model):
    """
    Checking the matching of the dress colour with the given one
    """
    img_h, img_w = img.shape[:2]
    dress_block_h = int(img_h//6.5)
    dress_block_w = int(img_w//3)
    dress_block = img[dress_block_h:dress_block_h*2, dress_block_w:dress_block_w*2]
    cv2.imshow("Dress", dress_block)
    dress_block = cv2.resize(dress_block, (32, 32), interpolation = cv2.INTER_AREA)
    dress_block = dress_block.astype("float32")/255.0
    dress_block = np.expand_dims(dress_block, axis=0)
    dress_match = dress_model.predict([dress_block, target_dress])
    if dress_match[0][0]>0.5:
        print("Dress match confidence: ", dress_match[0][0])
        dress_match = 1
    else:
        dress_match = 0
    return dress_match

def check_face(face_img, target_face_embeddings, facenet):
    """
    Checking the matching of the Face detected with the given one
    """
    face_img = cv2.resize(face_img, (224, 224), interpolation = cv2.INTER_AREA)
    face_img = np.expand_dims(face_img, axis=0).astype('float32')
    face_img = preprocess_input(face_img, version = 2)
    face_embedding = facenet.predict(face_img)
    scores = [cosine(face_embedding, em) for em in target_face_embeddings]
    target_person = np.argmax(scores)
    return target_person, scores[target_person]
    
def print_spotted(spot, miss, start, FPS, spotting_criteria):
    """
    Printing the servillence info(when spotted, what matched)
    """
    spot_times = []
    miss_times = []
    for i in range(len(spot)):
        
        tn = spot[i]//FPS
        second = int(tn%60)
        minute = int((tn//60)%60)
        hour = int((tn//60)//60)
        spot_times.append((hour, minute, second))

        tn = miss[i]//FPS
        second = int(tn%60)
        minute = int((tn//60)%60)
        hour = int((tn//60)//60)
        miss_times.append((hour, minute, second))
    start_h, start_m, start_s = map(int, start.split(':'))
    for i in range(len(spot)):
        if (miss_times[i][0]-spot_times[i][0])*3600+(miss_times[i][1]-spot_times[i][1])*60+(miss_times[i][2]-spot_times[i][2])>2:
            if spotting_criteria[i] == 1:
                spot_str = "Face Matched"
            else:
                spot_str = "Dress Matched"
            print()
            print(f"Person spotted from {start_h+spot_times[i][0]}:{start_m+spot_times[i][1]}:{start_s+spot_times[i][2]} to {start_h+miss_times[i][0]}:{start_m+miss_times[i][1]}:{start_s+miss_times[i][2]}: "+spot_str)
            print()

    

def face_rec(img, detector, target_face_embeddings, facenet): 
    """
    Detect the face from the person image and return the recognition info
    """
    boxes = detector.detect_faces(img)
    matchx = 0
    box = []
    person = 0
    if boxes:
        box = boxes[0]['box']
        conf = boxes[0]['confidence']
        x, y, w, h = box[0], box[1], box[2], box[3]
        if conf > 0.5:
            face_img = img[y:int(y+h), x:int(x+w)]
            person, person_conf = check_face(face_img, target_face_embeddings, facenet)
            if person_conf>0.4:
                print("Face match Confidence: ", person_conf)
                matchx = 1
    
    return box, matchx, person

def person_detect(new_frame, model, output_layers):
    """
    Detect the person from the given frame and return the boxes
    """
    height, width, channels = new_frame.shape
    blob, outputs = detect_objects(new_frame, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.9, 0.4)
    boxes = np.array(boxes)[indexes]
    boxes = np.clip(boxes, 0, 416)
    return boxes

def get_siamese_dress():
    """
    Get the siamese model for dress recognition
    """
    feature_extractor = get_feature_extractor()
    inpA = Input((32, 32, 3))
    inpB = Input((32, 32, 3))
    fetA = feature_extractor(inpA)
    fetB = feature_extractor(inpB)
    def euclidian_distance(vectors):
        featA, featB = vectors
        squred_sum = k.sum(k.square(featA-featB), axis=1, keepdims=True)
        distance = k.sqrt(k.maximum(squred_sum, k.epsilon()))
        return distance
    dist = Lambda(euclidian_distance)([fetA, fetB])
    outputs = Dense(1, activation='sigmoid')(dist)
    model = Model([inpA, inpB], outputs)
    return model

def start_video(video_path, target_faces, target_dress_path):
    """
    Main function which takes the video, target faces, target dress image and do the whole servillence to identify that person using its Face, Dress and record the time when he/she spotted
    """
    model, classes, colors, output_layers = load_yolo()
    dress_checker = get_siamese_dress()
    dress_checker.load_weights('model_dresses.h5')
    target_dress = cv2.imread(target_dress_path)
    target_dress = cv2.resize(target_dress, (32, 32), interpolation = cv2.INTER_AREA)
    target_dress = target_dress.astype("float32")/255.0
    target_dress = np.expand_dims(target_dress, axis=0)
    fs = 0
    init = False
    tracker = cv2.TrackerCSRT_create()
    spotted = []
    missing = []
    spotting = 0
    face_spotting = 0
    dress_spotting = 0
    spotting_criteria = []
    cap = cv2.VideoCapture(video_path)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    detector = MTCNN()
    facenet = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    target_face_embeddings = []
    for face_img in target_faces:
        face_img = cv2.resize(face_img, (224, 224), interpolation = cv2.INTER_AREA)
        face_img = np.expand_dims(face_img, axis=0).astype('float32')
        face_img = preprocess_input(face_img, version = 2)
        embedding = facenet.predict(face_img)
        target_face_embeddings.append(embedding)
    
    now = datetime.now()
    start_time = now.strftime("%H:%M:%S")
    while True:
        fs += 1
        if init:
            _, frame = cap.read()
            if frame is None:
                break
            frame = cv2.resize(frame, (416, 416))
            success, target_box = tracker.update(frame)
            
            if success:
                spotting = 1
                (x, y, w, h) = [int(v) for v in target_box]
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                bxs = person_detect(frame, model, output_layers)
                
                if len(bxs)>0:
                    if len(list(bxs.shape))==3:
                        bxs = np.squeeze(bxs, axis=1)
                    try:
                        bxs = sorted(bxs, key= lambda x: (x[0][2]*x[0][3]), reverse = True)
                        if len(bxs)>0:
                            bxs = bxs[0]
                            bxs = bxs.reshape((4, ))
                            img = frame[int(bxs[1]):int(bxs[1]+bxs[3]), int(bxs[0]):int(bxs[0]+bxs[2])]
                            bx, match, ps = face_rec(img, detector, target_face_embeddings, facenet)
                            if match == 1:
                                tracker = cv2.TrackerCSRT_create()
                                target_box = (int(bx[0]+bxs[0]), int(bx[1]+bxs[1]), int(bx[2]), int(bx[3]))
                                cv2.rectangle(frame, (target_box[0], target_box[1]), (target_box[0]+target_box[2], target_box[1]+target_box[3]), (0, 255, 0), 2)
                                success = tracker.init(frame, target_box)
                            else:
                                dress_match = check_dress(img, target_dress, dress_checker)
                            
                            if dress_match == 1 or match == 1:
                                if spotting==0:
                                    if match == 1:
                                        dress_spotting = 0
                                        face_spotting = 1
                                    else:
                                        face_spotting = 0
                                        dress_spotting = 1
                                    spotted.append(fs)
                                    spotting = 1
                                else:
                                    if match == 1 and dress_spotting == 1:
                                        missing.append(fs)
                                        spotting_criteria.append(2)
                                        spotted.append(fs)
                                        dress_spotting = 0
                                        face_spotting = 1
                                    elif match == 0 and face_spotting == 1:
                                        missing.append(fs)
                                        spotting_criteria.append(1)
                                        spotted.append(fs)
                                        dress_spotting = 1
                                        face_spotting = 0
                            else:
                                if spotting == 1:
                                    spotting = 0
                                    missing.append(fs)
                                    if face_spotting == 1:
                                        spotting_criteria.append(1)
                                    else:
                                        spotting_criteria.append(2)
                            cv2.rectangle(frame, (int(bxs[0]), int(bxs[1])), (int(bxs[0]+bxs[2]), int(bxs[1]+bxs[3])), (255, 0, 0), 2)
                    except:
                        bxs = sorted(bxs, key= lambda x: (x[2]*x[3]), reverse = True)
                        if len(bxs)>0:
                            bxs = bxs[0]
                            bxs = bxs.reshape((4, ))
                            img = frame[int(bxs[1]):int(bxs[1]+bxs[3]), int(bxs[0]):int(bxs[0]+bxs[2])]
                            bx, match, ps = face_rec(img, detector, target_face_embeddings, facenet)
                            if match == 1:
                                tracker = cv2.TrackerCSRT_create()
                                target_box = (int(bx[0]+bxs[0]), int(bx[1]+bxs[1]), int(bx[2]), int(bx[3]))
                                cv2.rectangle(frame, (target_box[0], target_box[1]), (target_box[0]+target_box[2], target_box[1]+target_box[3]), (0, 255, 0), 2)
                                success = tracker.init(frame, target_box)
                            else:
                                dress_match = check_dress(img, target_dress, dress_checker)
                            
                            if dress_match == 1 or match == 1:
                                if spotting==0:
                                    if match == 1:
                                        dress_spotting = 0
                                        face_spotting = 1
                                    else:
                                        face_spotting = 0
                                        dress_spotting = 1
                                    spotted.append(fs)
                                    spotting = 1
                                else:
                                    if match == 1 and dress_spotting == 1:
                                        missing.append(fs)
                                        spotting_criteria.append(2)
                                        spotted.append(fs)
                                        dress_spotting = 0
                                        face_spotting = 1
                                    elif match == 0 and face_spotting == 1:
                                        missing.append(fs)
                                        spotting_criteria.append(1)
                                        spotted.append(fs)
                                        dress_spotting = 1
                                        face_spotting = 0
                            else:
                                if spotting == 1:
                                    spotting = 0
                                    missing.append(fs)
                                    if face_spotting == 1:
                                        spotting_criteria.append(1)
                                    else:
                                        spotting_criteria.append(2)
                            cv2.rectangle(frame, (int(bxs[0]), int(bxs[1])), (int(bxs[0]+bxs[2]), int(bxs[1]+bxs[3])), (255, 0, 0), 2)
                    
                
        else:
            
            _, frame = cap.read()
            frame = cv2.resize(frame, (416, 416))
            bxs = person_detect(frame, model, output_layers)
            bxs = sorted(bxs, key= lambda x: (x[0][2]*x[0][3]), reverse = True)
            if len(bxs)>0:
                bxs = bxs[0]
                bxs = bxs.reshape((4, ))
                img = frame[int(bxs[1]):int(bxs[1]+bxs[3]), int(bxs[0]):int(bxs[0]+bxs[2])]
                bx, match, ps = face_rec(img, detector, target_face_embeddings, facenet)
                dress_match = 0
                if match == 1:
                    init = True
                    tracker = cv2.TrackerCSRT_create()
                    target_box = (int(bx[0]+bxs[0]), int(bx[1]+bxs[1]), int(bx[2]), int(bx[3]))
                    cv2.rectangle(frame, (target_box[0], target_box[1]), (target_box[0]+target_box[2], target_box[1]+target_box[3]), (0, 255, 0), 2)
                    success = tracker.init(frame, target_box)
                else:
                    dress_match = check_dress(img, target_dress, dress_checker)
                
                if dress_match == 1 or match == 1:
                    if spotting==0:
                        if match == 1:
                            dress_spotting = 0
                            face_spotting = 1
                        else:
                            face_spotting = 0
                            dress_spotting = 1
                        spotted.append(fs)
                        spotting = 1
                    else:
                        if match == 1 and dress_spotting == 1:
                            missing.append(fs)
                            spotting_criteria.append(2)
                            spotted.append(fs)
                            dress_spotting = 0
                            face_spotting = 1
                        elif match == 0 and face_spotting == 1:
                            missing.append(fs)
                            spotting_criteria.append(1)
                            spotted.append(fs)
                            dress_spotting = 1
                            face_spotting = 0
                else:
                    if spotting == 1:
                        spotting = 0
                        missing.append(fs)
                        if face_spotting == 1:
                            spotting_criteria.append(1)
                        else:
                            spotting_criteria.append(2)


                cv2.rectangle(frame, (int(bxs[0]), int(bxs[1])), (int(bxs[0]+bxs[2]), int(bxs[1]+bxs[3])), (255, 0, 0), 2)
        if frame is not None:
            cv2.imshow("Image", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    print_spotted(spotted, missing, start_time, FPS, spotting_criteria)
    
    return spotted, missing




if __name__ == "__main__":
    video_path = input("Provide video path: \n")
    image_path = input("Provide face image path: \n")
    target_dress_path = input("Provide dress image path: \n")
    image = cv2.imread(image_path)
    images = [image]
    spotted, missing = start_video(video_path, images, target_dress_path)
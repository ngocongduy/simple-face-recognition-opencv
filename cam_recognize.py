# import OpenCV module
import cv2
# import os module for reading training data directories and paths
import os
# import numpy to convert python lists to numpy arrays as
# it is needed by OpenCV face recognizers
import numpy as np


# ### Training Data

# The more images used in training the better. Normally a lot of images are used for training a face recognizer so that it can learn different looks of the same person, for example with glasses, without glasses, laughing, sad, happy, crying, with beard, without beard etc. To keep our tutorial simple we are going to use only 12 images for each person.
#
# So our training data consists of total 2 persons with 12 images of each person. All training data is inside _`training-data`_ folder. _`training-data`_ folder contains one folder for each person and **each folder is named with format `sLabel (e.g. s1, s2)` where label is actually the integer label assigned to that person**. For example folder named s1 means that this folder contains images for person 1. The directory structure tree for training data is as follows:
#
# ```
# training-data
# |-------------- s1
# |               |-- 1.jpg
# |               |-- ...
# |               |-- 12.jpg
# |-------------- s2
# |               |-- 1.jpg
# |               |-- ...
# |               |-- 12.jpg
# ```

class Predictor():
    # function to detect face using OpenCV
    def _detect_face(self, img, scaleFactor = 1.01):
        # convert the test image to gray image as opencv face detector expects gray images
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # let's detect multiscale (some images may be closer to camera than others) images
        # result is a list of faces
        print("Scale factor {}".format(scaleFactor))
        faces = self.face_cascade.detectMultiScale(gray,
                                                   scaleFactor=scaleFactor,
                                                   minSize=(120, 120),  # ignore smaller object
                                                   minNeighbors=5);

        # if no faces are detected then return original img
        if (len(faces) == 0):
            return None, None

        # under the assumption that there will be only one face,
        # extract the face area
        (x, y, w, h) = faces[0]

        # return only the face part of the image
        return gray[y:y + w, x:x + h], faces[0]

    def _prepare_training_data(self, data_folder_path):
        # ------STEP-1--------
        # get the directories (one directory for each subject) in data folder
        dirs = os.listdir(data_folder_path)

        face_list = []
        label_list = []
        # let's go through each directory and read images within it
        for dir_name in dirs:

            # our subject directories start with letter 's' so
            # ignore any non-relevant directories if any
            if not dir_name.startswith("s"):
                continue;

            # ------STEP-2--------
            # extract label number of subject from dir_name
            # format of dir name = slabel
            # , so removing letter 's' from dir_name will give us label
            label = int(dir_name.replace("s", ""))

            # build path of directory containin images for current subject subject
            # sample subject_dir_path = "training-data/s1"
            subject_dir_path = data_folder_path + "/" + dir_name

            # get the images names that are inside the given subject directory
            subject_images_names = os.listdir(subject_dir_path)

            # ------STEP-3--------
            # go through each image name, read image,
            # detect face and add face to list of faces
            for scale in self.scales_for_tunning:
                faces = []
                labels = []
                for image_name in subject_images_names:

                    # ignore system files like .DS_Store
                    if image_name.startswith("."):
                        continue;

                    # build image path
                    # sample image path = training-data/s1/1.pgm
                    image_path = subject_dir_path + "/" + image_name

                    print("Image name to be read " + image_path)

                    # read image
                    image = cv2.imread(image_path)

                    # display an image window to show the image
                    cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
                    cv2.waitKey(100)

                    # detect face
                    face, rect = self._detect_face(image, scale)

                    # ------STEP-4--------
                    # for the purpose of this tutorial
                    # we will ignore faces that are not detected
                    if face is not None:
                        # add face to list of faces
                        faces.append(face)
                        # add label for this face
                        labels.append(label)
                face_list.append(faces)
                label_list.append(labels)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()

        max_face_detected = 0
        max_index = -1
        # print(face_list)
        print(label_list)
        for i in range(len(face_list)):
            if max_face_detected < len(face_list[i]):
                max_face_detected = len(face_list[i])
                max_index = i
        if max_index >= 0:
            print("Best scale is {} with number of face is {} ".format(max_index, max_face_detected))
            return face_list[max_index], label_list[max_index]
        else:
            return None, None

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
        self.subjects = ["", "abcxyz"]
        self.scales_for_tunning = [1.01 + i*0.01 for i in range(0,10)]
        # ### Train Face Recognizer
        # 1. EigenFace Recognizer: This can be created with `cv2.face.createEigenFaceRecognizer()`
        # 2. FisherFace Recognizer: This can be created with `cv2.face.createFisherFaceRecognizer()`
        # 3. Local Binary Patterns Histogram (LBPH): This can be created with `cv2.face.LBPHFisherFaceRecognizer()`
        # Pre-trained face_recognizer
        pretrained_recognizer_path = "trainner/trainner.yml"
        is_pretrained_exist = os.path.exists(pretrained_recognizer_path)
        print("pretrained model existed " + str(is_pretrained_exist))
        if not is_pretrained_exist:

            print("Train a new model!")
            print("Preparing data...")
            faces, labels = self._prepare_training_data("images")
            print("Data prepared")

            # print total faces and labels
            print("Total faces: ", len(faces))
            print("Total labels: ", len(labels))

            # create our LBPH face recognizer
            face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            # or use EigenFaceRecognizer by replacing above line with
            # face_recognizer = cv2.face.EigenFaceRecognizer_create()

            # or use FisherFaceRecognizer by replacing above line with
            # face_recognizer = cv2.face.FisherFaceRecognizer_create()

            # train our face recognizer of our training faces
            face_recognizer.train(faces, np.array(labels))
            face_recognizer.save(pretrained_recognizer_path)

        else:
            print("Load existing model")
            face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            face_recognizer.read(pretrained_recognizer_path)
        self.face_recognizer = face_recognizer

    # given width and heigh
    def _draw_rectangle(self, img, rect):
        (x, y, w, h) = rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # function to draw text on give image starting from
    # passed (x, y) coordinates.
    def _draw_text(self, img, text, x, y):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    def predict(self, test_img):
        # make a copy of the image as we don't want to chang original image
        img = test_img.copy()
        # detect face from the image
        face, rect = self._detect_face(img)
        if face is not None or not rect is None:
            # predict the image using our face recognizer
            label, confidence = self.face_recognizer.predict(face)

            print(label)
            print(confidence)

            # get name of respective label returned by face recognizer
            label_text = self.subjects[label]

            # draw a rectangle around face detected
            self._draw_rectangle(img, rect)
            # draw name of predicted person
            self._draw_text(img, label_text, rect[0], rect[1] - 5)
        return img

def predict_some_test_images():
    print("Predicting images...")

    test_image_folder = "test-images"
    test_images = os.listdir(test_image_folder)
    predictor = Predictor()
    for image_name in test_images:
        print("Try to guess image " + image_name)
        image_path = os.path.join(test_image_folder, image_name)
        test_image = cv2.imread(image_path)
        guess = predictor.predict(test_image)
        print(guess.shape)
        print("Predicted")
        cv2.imshow("abcxyz", cv2.resize(guess, (480, 640)))
        cv2.waitKey(500)

def runtime_webcam_guess():
    def __clean_up():
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()

    webcam = cv2.VideoCapture(0)
    predictor = Predictor()
    while True:
        try:
            check, frame = webcam.read()
            print("Is webcam running " + str(check))  # prints true as long as the webcam is running
            # print(frame) #prints matrix values of each frame
            # cv2.imshow("Capturing", frame)

            # Pass the frame to detect
            guess = predictor.predict(frame)
            print("Predicted")
            print("Type of predicted frame:" + str(type(guess)))
            print(guess.shape)
            cv2.imshow(predictor.subjects[1], guess)

            # waitKey(1) will display a frame for 1 ms, after which display will be automatically closed
            key = cv2.waitKey(5)

            if key == ord('q'):
                __clean_up()
                break

        except KeyboardInterrupt:
            __clean_up()
            break

predict_some_test_images()
runtime_webcam_guess()
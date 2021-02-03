import os

import cv2


count = 0
my_imgs_dir = "images/s1"
imgs = os.listdir(my_imgs_dir)
nums = []
for ele in imgs:
    prefix = "saved_img_"
    if prefix in ele:
        parts = ele.split(prefix)
        try:
            last_count = int(parts[1][0])
            nums.append(last_count)
        except:
            pass
if len(nums) > 0:
    count = max(nums)

print("About to save images with index start from {}".format(count+1))

while True:
    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(0)
    try:
        check, frame = webcam.read()
        print(check) #prints true as long as the webcam is running
        # print(frame) #prints matrix values of each framecd
        cv2.imshow("Capturing", frame)
        # waitKey(1) will display a frame for 1 ms, after which display will be automatically closed
        key = cv2.waitKey(1)
        if key == ord('s'):
            count+=1
            original_file_name = os.path.join(my_imgs_dir,'saved_img_{}.jpg'.format(count))
            print("Saving original image with name: " + original_file_name)
            cv2.imwrite(filename=original_file_name, img=frame)
            webcam.release()

            img_new = cv2.imread(original_file_name, cv2.IMREAD_GRAYSCALE)
            img_new = cv2.imshow("Captured Image", img_new)
            cv2.waitKey(1650)
            cv2.destroyAllWindows()

            print("Processing image with name..." + original_file_name)
            img_ = cv2.imread(original_file_name, cv2.IMREAD_ANYCOLOR)
            print("Converting RGB image to grayscale...")
            gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
            print("Converted RGB image to grayscale...")
            print("Resizing image to 28x28 scale...")
            img_ = cv2.resize(gray,(28,28))
            print("Resized...")
            resized_file_name = os.path.join(my_imgs_dir, 'saved_img-28x28_{}.jpg'.format(count))
            img_resized = cv2.imwrite(filename=resized_file_name, img=img_)
            print("Image saved!")
        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break

    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break
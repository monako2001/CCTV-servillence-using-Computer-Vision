# CCTV-servillence-using-Computer-Vision
A program to check a CCTV footage and identifying a person using his/her Face, Dress and recording time of appearance and disappearance

## Objective:
Crime investigators often use to check some CCTV footage of shops, mall, roads, schools or office to find out the criminal. Sometimes the footage is very long and it becomes difficult to check it out. Also if there is a missing case they should be also interested in online servillence of some railway station, bus stop etc. Only clue they have are the face of the victim and an image of his/her dress. 
Our aim is to help them in the servillence for the missing one or the criminal.
<br>

## Approach:
- The program takes the path to the CCTV footage. In case of online servillence it takes the ip address. It also takes the Face image and dress image of the target person.
- In each frame it first detects the persons using YOLOv3 object detection model.
- For each person it detects its face using MTCNN face detector and calculates the matching score with the given one. If the face matches it records the time and tracks the face. For the face matching it uses pretrained VGGFace.
- If the face does not match it checks out the dress color is matching or not. For getting the shirt colour we have cut out a portion from its back. To get the required portion we have used some basic principles of Human anatomy. If the dress colour matches it records the time.
- When it can't find the person it also records the time.
- In this way the program goes through the whole video and records the time. At last it prints all of them out.
<br>

## Innovation:
- In most of the CCTV footage the picture is not much clear which makes face recognition very difficult. So we have also used the dress colour as a factor to find out the target person. Dress information is easily available for a missing person and investigators use it to find the person. Two person can have similar dresses but atleast this program can filter out those persons from the footage to make the work easier.
<br>

## Testing:
- First install all the dependencies from the `requirements.txt` file. It is preferred to use virtual environment.
- Download the YOLOv3 weights file from [here](https://pjreddie.com/darknet/yolo/). Here we have used the model of 416x416 image size.
- Keep the weights file in the same directory as the file `person_detect_final.py`.
- Now run the `person_detect_final.py` and give it the required paths for images and video.
- For testing initially you can use the example files provided.
- when the running ends it will print out the servillence informations.

## References
- [YOLO Object Detection using OpenCV](https://towardsdatascience.com/yolo-object-detection-with-opencv-and-python-21e50ac599e9)
- [Face Detection using MTCNN](https://www.mygreatlearning.com/blog/real-time-face-detection/)
- [Face recognition using pretrained Facenet](https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/)
- [Dress recognition dataset](https://medium.com/data-science-insider/clothing-dataset-5b72cd7c3f1f)
- [Training Siamese network for dresses](https://www.kaggle.com/mainak2001/training-siamese-network-for-dresses)

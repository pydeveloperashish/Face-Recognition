This is a step by step explanation of how train and run Face Recognizer.


To watch the tutorial video, here is the link of Neha Yadav video in her channel:- https://www.youtube.com/watch?v=h21wMKGs0qs

1) Make a folder named as 'train-images' where you will store images to train.

2) Make 1 or Multiple folders under train-images folder named as 0,1,2 and so on to store the images of people you want to recognize.

3) For every person's images, different folders will be used. For example for person A, his images will be stored in folder named as 0
and images of Person B, his images will be stored in folder named as 1 and so on.

4) Open faceRecognition.py and Give path to haar classifier as i have given in faceDetection function.

5) Now open Create_dataset_from_webcam.py to capture images from your webcame if you want to create dataset for your own images. It will click multiple images from webcam till you end this program. 
Make sure it clicks atleast 300 images before you end the program. Do the required configuration as i have mentioned in comments.

6) Your images will be stored at train-images/0/ folder

7) Now at train-images/1/ folder, you can store images of any other person you want to recognise.

8) Now open train_model.py to train your model to recognize whoever face you want. Do the required configuration.

9) in train_model.py, If you want to recognize only one person then write:- name={0:"name"} thats all. Dont write for id number 1.

10) Now your training will get done and trainingData.yml file will be saved.

11) Now open load_model_image.py to recognize face from an image. Do the required configurations.

12) Now open load_model_video.py to recognize face from a video or webcam. Do the required configurations.

13) If you still face difficulty then either watch that video or message me in Instagram:- @py_developer.ashish

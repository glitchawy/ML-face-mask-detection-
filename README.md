# ML-face-mask-detection-
Face Mask detection using ML manually trained model with Modded DNN 

HOW TO USE :

/----------------------------------- train.py--------------------------------------------------------\
- Change your initial learning rate ( INIT_LR , EPOCHS , BS ) line23 train.py ( keep if you are not aware )
- Change directory line30 train.py ( put your dataset path )
- Change categories line31 train.py ( put your dataset file names that already named as your detection classess )
- Change model name line111 train.py ( if nedded )
- Change Output Plot name line125 train.py ( if nedded )
- run train.py file and wait for training to end .......


/---------------------------------- detect_mask.py -------------------------------------------------\
- change line48 the number 0 to another ( if you want to change the predection acoording to the faces count )
- change line100 according to step4 in the train.py "how to use"
- change line118 if you want to change labeling



NOTES :
 THIS TRAINING CODE AND WEIGHTS PERFECTLY FITS DNN PHOTO PROCESSING AND TRAINING , SO YOU CAN USE IT FOR ANOTHER PORPUSE FOR DETECTING MORE CATEGOERIES AND MORE OBJECT BY JYST FEEDING IT WITH THE RIGHT DATASETS ...
 
 HOW DOES IT REALLY WORKS :
 - preproccessing the dataset photos
 - initializitng the MOBILENETV2 DNN for feeding it
 - Compilimg the model
 - Training the head
 - Network evaluation
 - Saving the model 
 - Drawing a plot for monitoring accuracy
 - loop over the detections
 - convert it from BGR to RGB channel and ordering, resize
 - bounding boxes to their respective lists
 - load our serialized face detector model from disk
 - load the face mask detector model from disk
 - loop over the frames from the video stream
 - unpack the bounding box and predictions
 - draw bounding box and text
 - display the label and bounding box rectangle on the output
 - some keras optimization
 - voala ... it works
 
 --------------------------------------------------  WARNING ---------------------------------------------------
 HAVING A GPU MAKES EVERYTHING GOES FAST ...... HAVING ERRORS ON THE KERAS LIBRARY IN THE INCLUDE ISN'T AN ISUUE IT
 WILL JUST HAPPEN IF YOU DON'T HAVE GPU

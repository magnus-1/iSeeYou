# iSeeYou
You will need to install eigen and opencv3, tested with eigen3.3.1 and opencv 3.1.0-dev without -lopencv_ts there was a bug in september.

g++ -std=gnu++14 -I/usr/local/include/eigen3/ -lopencv_calib3d -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_video -lopencv_videoio -lopencv_videostab -I/usr/local/opt/opencv3/include -L/usr/local/opt/opencv3/lib  *.cpp -o prog 

To run program:  
enter: ./prog

to run the testcases for the nn: 

enter:  ./prog -testcase1

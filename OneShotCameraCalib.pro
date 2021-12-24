TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp \
        src/ChessboardFromCorners.cpp \
        src/CornerDetection.cpp


INCLUDEPATH += /usr/local/include/opencv4
LIBS += -L/usr/local/lib \
-lopencv_calib3d \
-lopencv_core \
-lopencv_dnn \
-lopencv_features2d \
-lopencv_flann \
-lopencv_gapi \
-lopencv_highgui \
-lopencv_imgcodecs \
-lopencv_imgproc \
-lopencv_ml \
-lopencv_objdetect \
-lopencv_photo \
-lopencv_stitching \
-lopencv_video \
-lopencv_videoio \

HEADERS += \
    include/ChessboardFromCorners.h \
    include/CornerDetection.h

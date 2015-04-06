########################################
#                                      #
#    Makefile for the NCV Projects.    #
#                                      #
########################################

PREFIX=/home/bruno/Lib
OPENCV_PREFIX=$(PREFIX)/opencv-2.4.10/local

# OpenCV #
OPENCV_INC_DIR = -I$(OPENCV_PREFIX)/include
OPENCV_LIB_DIR = -L$(OPENCV_PREFIX)/lib
OPENCV_LINK_FLAGS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_flann -lopencv_nonfree
##########

BIN_DIR = ./bin
SRC_DIR = ./src
INC_DIR = ./include
OBJ_DIR = $(SRC_DIR)/.lib

INCLUDES = -I$(INC_DIR) $(OPENCV_INC_DIR)
LDFLAGS = $(OPENCV_LIB_DIR)
LFLAGS = $(OPENCV_LINK_FLAGS)

$(BIN_DIR)/match_cover: $(SRC_DIR)/match_cover.cpp $(OBJ_DIR)/geometry_utils.o $(OBJ_DIR)/visualization_utils.o
	$(CXX) $(SRC_DIR)/match_cover.cpp $(OBJ_DIR)/geometry_utils.o $(OBJ_DIR)/visualization_utils.o \
	       $(LDFLAGS) $(CFLAGS) $(INCLUDES) $(LFLAGS) -o $(BIN_DIR)/match_cover

$(OBJ_DIR)/geometry_utils.o: $(SRC_DIR)/geometry_utils.cpp $(INC_DIR)/geometry_utils.h
	$(CXX) -c $(SRC_DIR)/geometry_utils.cpp $(INCLUDES) -o $(OBJ_DIR)/geometry_utils.o

$(OBJ_DIR)/visualization_utils.o: $(SRC_DIR)/visualization_utils.cpp $(INC_DIR)/visualization_utils.h
	$(CXX) -c $(SRC_DIR)/visualization_utils.cpp $(INCLUDES) -o $(OBJ_DIR)/visualization_utils.o 

CXX ?= g++
BIN_DIR := ../../bin-linux
LIB_DIR := ../../lib-so

CXXFLAGS := -Wall -O2 -g -std=c++17 -fPIC -pthread

.PHONY: all
all: $(BIN_DIR)/surdata_transfer_selftest $(LIB_DIR)/libsurdata_transfer.so

$(BIN_DIR)/surdata_transfer_selftest: surdata_transfer.cc surdata_transfer_selftest.cc surdata_transfer.h
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ surdata_transfer.cc surdata_transfer_selftest.cc

$(LIB_DIR)/libsurdata_transfer.so: surdata_transfer.cc surdata_transfer.h
	mkdir -p $(LIB_DIR)
	$(CXX) $(CXXFLAGS) -shared -o $@ surdata_transfer.cc

all:
	mkdir -p objs
	nvcc -g -G rgbdodometry.cu -c -o objs/rgbdodometry.o -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_contrib -lopencv_calib3d -lopencv_gpu
	g++ main.cpp -c -o objs/main.o -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_contrib -lopencv_calib3d -lopencv_gpu -I/usr/local/cuda-6.0/include
	g++ -Wall -o rgbd.out objs/main.o objs/rgbdodometry.o -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_contrib -lopencv_calib3d -lopencv_gpu -lcudart -L/usr/local/cuda-6.0/lib

clean:
	rm -rf objs/
	rm rgbd.out

run1:
	./rgbd.out rgbdodometry/image_00000.png rgbdodometry/depth_00000.png rgbdodometry/image_00002.png rgbdodometry/depth_00002.png

run2:
	./rgbd.out rgbdodometry/snapshot0.jpg rgbdodometry/d_snapshot0.jpg rgbdodometry/snapshot1.jpg rgbdodometry/d_snapshot1.jpg

all:
	nvcc -O3 -arch=sm_86 ${CUFILES} -o ${EXECUTABLE} 
clean:
	rm -f *~ *.exe
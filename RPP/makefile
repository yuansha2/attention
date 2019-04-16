main.exe: main.o mic_model.o mic_model.h
	g++ main.o mic_model.o -o main.exe
main.o: main.cpp mic_model.h
	g++ -c main.cpp -o main.o
mic_model.o: mic_model.cpp mic_model.h
	g++ -c mic_model.cpp -o mic_model.o
clean:
	rm *.o *.exe

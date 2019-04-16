main.exe: main.o CART.o CART.h Node.o Node.h
	g++ main.o CART.o Node.o -o main.exe -std=c++11 
main.o: main.cpp Node.h CART.h
	g++ -c main.cpp -o main.o -std=c++11 
CART.o: CART.cpp CART.h
	g++ -c CART.cpp -o CART.o -std=c++11 
Node.o: Node.cpp Node.h
	g++ -c Node.cpp -o Node.o -std=c++11 
clean:
	rm *.o *.exe

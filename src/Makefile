all:
	g++ -Ofast -march=native -O4 -Wall -fopenmp -std=c++0x -I eigen reps.cc -o reps

run:
	./reps --dim=300 --epohs=2 --model=model -l=1 --alpha=0.01 --source=books --target=electronics

batch:
	./reps --dim=100 --epohs=100 --model=model.final -l=100 --alpha=0.1 --source=$(src) --target=$(tgt)

clean:
	rm -f reps
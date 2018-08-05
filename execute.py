import subprocess;
import multiprocessing as mp;


def function(name):
	subprocess.call(name);

def multi(list):
	p = mp.Pool(mp.cpu_count());
	p.map(function,list);
	p.close();

if __name__ == "__main__":

	REPS = [ i for i in range(0,30)];
	list = [];
	for trial in REPS:
			list.append(["python","GA.py",str(trial)]); 								
			
#	function(list[0]);
	multi(list);
	
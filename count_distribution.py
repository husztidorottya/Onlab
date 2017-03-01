import sys
from collections import defaultdict

distribution = defaultdict(int)

for line in sys.stdin:

	if line!="\n":	
		number_of_classes = line.count("\t")
		distribution[number_of_classes] += 1

for key,value in distribution.items():
	sys.stdout.write('{}:  {}\n'.format(key,value))

#
#két szófajú szavaknak az eloszlását számolja ki, hogy miből mibe vált
#

import sys
from collections import defaultdict

distribution = defaultdict(lambda : defaultdict(int))

for line in sys.stdin:

	if line!="\n":
		splited_line = line.split("\t")

		j=0
		class_one = ""
		class_two = ""
		for i in splited_line:
			if j==1:
				class_one = i
			if j==2:
				class_two = i
			j+=1	
	
		distribution[class_one][class_two] +=1
	

for key,value in distribution.items():
	for k,v in value.items():
		sys.stdout.write('{0}\t{1}\t{2}\n'.format(key,k,v))

#
# Kiszedi csak a szóalakokat a fájlból
#

import sys

infilename = input('Input file:')
outfilename = input('Output file:')

file = open(outfilename,'w')
file.close()

with open(infilename) as inputfile:
	for line in inputfile:
		if line!='\n':
			splited_line = line.split('\t')
			
			with open(outfilename,'a') as outputfile:
				outputfile.write(splited_line[0] + '\n')
		

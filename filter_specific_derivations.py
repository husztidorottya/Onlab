#
# Kiszűri a megadott derivációláncnak megfelelő sorokat. A derivációk megadásának módja: tab/Ntab/N
#

import re
import sys

derivations = input('Derivaciok:')
derivations = derivations.replace('\t','[^0-9/]*')
patt = re.compile(derivations)

filename = input('Bementei fajl:')
output_filename = input('Kimeneti fajl:')

output_file = open(output_filename,'w')
output_file.close()

previous_empty_line = True

with open(filename) as input_file:
	for line in input_file:
		if line!='\n' and previous_empty_line:
			if line.find('/')!=-1:
				m = patt.match(line)
				if m:
					splited_line = line.replace('[',';;;').replace(']',';;;').replace('\t',';;;').split(';;;')
					final_line = []

					for i in range(0,len(splited_line)):
						if i==0:
							final_line.append(splited_line[i])
						else:
							if splited_line[i].find('/')!=-1:
								final_line.append(splited_line[i])
							else:
								if (i+1) < len(splited_line) and splited_line[i+1].find("/")!=-1:
									final_line.append(splited_line[i])
					with open(output_filename, 'a') as output_file:
						output_file.write('\t'.join(final_line) + '\n')
						output_file.write('\n')

				previous_empty_line = False

		if line=='\n':
			#with open(output_filename, 'a') as output_file:
			#	output_file.write('\n')
			previous_empty_line = True		
	

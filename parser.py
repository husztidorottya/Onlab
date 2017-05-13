#
# Kiszűri a megadott derivációláncnak megfelelő sorokat. A derivációk megadásának módja: tab/N;tab/N;
#

import re
import sys

filename = input('Bementei fajl:')
output_filename = input('Kimeneti fajl:')

output_file = open(output_filename,'w')
output_file.close()

previous_empty_line = True

with open(filename) as input_file:
	for line in input_file:
		if line!='\n' and previous_empty_line and line.find('/')!= -1:
			with open(output_filename, 'a') as output_file:
				output_file.write(line)
				output_file.write('\n')

			previous_empty_line = False

		if line=='\n':
			previous_empty_line = True		
	

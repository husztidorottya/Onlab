import sys

#blokk kezdet-e
previous_empty_line=1

for line in sys.stdin:

	if line!="\n" and previous_empty_line==1:
		parsed_line = []

		#melyik blokkban tartunk soron belül
		part_of_line=0

		#per jel volt elozo karakter vagy sem
		previous_character=0

		for c in line:

			if part_of_line==0:
				if c!="\t":
					parsed_line.append(c)
				else:
					part_of_line+=1	
			else:
				if part_of_line==1:
					if previous_character==1:
						parsed_line.append("\t")						
						parsed_line.append(c)
						previous_character=0
					else:
						if c=="/":
							previous_character=1

		for c in parsed_line:
			sys.stdout.write(c)

		sys.stdout.write("\n")
		previous_empty_line=0

	if line=="\n":
		sys.stdout.write("\n")
		previous_empty_line=1

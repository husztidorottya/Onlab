import sys

previous_empty_line = True

for line in sys.stdin:
	if line!="\n" and previous_empty_line:
		if line.find('/')!=-1:
			#split tab mentén
			splited_line = line.split("\t")

			final_line = []
			final_line.append(splited_line[0])	

			splited_line = str(splited_line).replace('[',';;;').replace(']',';;;').replace('|',';;;').split(";;")
			splited_line = str(splited_line).replace('/','///').split("//")
	

			for i in splited_line: 
				if i[0]=="/":
					for j in i:
						if j!="'":
							final_line.append(j)
						else:
							break
			final_line = eval(str(final_line).replace('/','\t'))			

			for i in final_line:
				sys.stdout.write(i)
		
			sys.stdout.write("\n")
			previous_empty_line=False
	if line=="\n":
		sys.stdout.write("\n")
		previous_empty_line=True


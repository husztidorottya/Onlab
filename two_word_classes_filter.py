import sys

for line in sys.stdin:
	if line!="\n" and line.count("\t")==2:
		sys.stdout.write('{}\n'.format(line))

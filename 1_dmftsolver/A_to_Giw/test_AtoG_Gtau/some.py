import subprocess as sub
U = 2.00

hey = f'{U:.2f}.txt'
myint = open(f'{U:.2f}.txt')
#myout = open('hey.txt')
p = sub.Popen('./docc', stdin=myint)#, stdout=myout)
p.wait()
#myouy.flush()

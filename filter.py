sizes = [500,1000,2000,4000,6000,8000,10000]
methods = ['BT','TR','LR$T_i^d$','TR$T_i^d$','LR$T_i^{sx}$','TR$T_i^{sx}$','LR$T_i^l$','TR$T_i^l$','LR$T_i^{sxl}$','TR$T_i^{sxl}$']  

head = ""
for x in sizes:
    head = head +  " & " + str(x) 
head = head + "\\\\\\hline\n"

def finish(net,averloglike,aversize):
    fileo.write("\n\\begin{table}\n \\begin{center}\n \\begin{tabular}{lrrrrrrr}\n")
    
    fileo.write(head)
    for i in range(len(methods)):
        fileo.write(methods[i])
        for j in sizes:
            fileo.write(" & " + f"{averloglike[i,j]:.{3}f}")
        fileo.write("\\\\\\hline\n")

    fileo.write("\\end{tabular}\n")
    fileo.write("\\end{center}\n")

    fileo.write("\\caption{Average log-likelihood for network "+ net.upper() + " }\n")
    fileo.write("\\label{"+net + "ll}\n")
    fileo.write("\\end{table}\n\n")

    fileo.write("\n\\begin{table}\n\\begin{center}\n\\begin{tabular}{lrrrrrrr}\n")
    fileo.write(head)
    for i in range(len(methods)):
        fileo.write(methods[i])
        for j in sizes:
            fileo.write(" & " + f"{aversize [i,j]:.{3}f}")
        fileo.write("\\\\\\hline\n")

       
    fileo.write("\\end{tabular}\n")
    fileo.write("\\end{center}\n")
    fileo.write("\\caption{Average size for network "+ net.upper() + " }\n")
    fileo.write("\\label{"+net + "si}\n")

    fileo.write("\\end{table}\n")

    return 

input = 'output6'
output = input + '.tex'
filei = open(input,'r')
fileo = open(output,"w")

lines = filei.readlines()



current = ''
averloglike = dict()
aversize = dict()
change = False
current = ''

for line in lines:
    if line[0] == '*':
        line = line[1:]
        (net,ssize) = line.split(',')
        (net,ext) = line.split('.')
        net = net.capitalize()
        
        if change and not current==net:
            change = False
            if current:
                finish(current,averloglike,aversize)
        current = net
    elif line[0] == "$":
        change = True

        line = line[1:]
        (n,ll,si) = line.split(",")
        averloglike[(int(n),int(ssize))] = float(ll)
        aversize[(int(n),int(ssize))]= float(si)
    
finish(current,averloglike,aversize)

        

            



var1 = "le"
var2 = "ro"
var3 = "ring"

var1, var2, var3 = var2 , var1, var3
var2 , var1, var3 = var1, var2, var3

for i in range(100):
    print(var1+var2, end='')
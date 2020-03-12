f = open("times_long.txt")
contents = f.read()
print(len(contents))
split1 = contents.split("\n")
print(len(split1))
split1 = [s.replace(" ", "") for s in split1]
split2 = [s.split(",") for s in split1]
split2 = split2[:-1]
split2.sort(key = lambda x : int(x[0]) * 30 + int(x[1]))
#print(split2)
print(len(split2))

for i in range(7):
  counter = 0
  time = 0
  for j in range(100):
    #print(split2[i * 30 + j][2])
    time += float(split2[i * 100 + j][2])
    if split2[i * 100 + j][3] == 'True':
      counter += 1
  print(i, 'time: ', time/100, 'counter:', counter)

doc = open('doc.txt','r')
fortest = open('fortest.txt','w')
cnt = 0
for line in doc:
    if len(line.split(" ")) >=2:
        cnt += 1
        fortest.write(line)
    if cnt >= 10000:
        break

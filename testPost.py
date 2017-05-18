import requests
import json

#file1 = {'text': open('fortest.txt', 'rb')}
#ms = requests.post('http://10.60.118.70:5000/ngramfile/',files=file1)

file2 = {'text': open('data/movieDataset_final_2w.txt', 'rb')}
ms = requests.post('http://10.60.118.70:9898/ngramfile/',files=file2)


#print ms.content

#text = "hello world\nwhat a beautiful day\nwhat a beautiful girl"
#ms = requests.post('http://10.60.118.70:5000/ngramfile/',headers={'Content-Type': 'application/json'},data=json.dumps({"text":text}))
#ms = requests.post('http://0:5000/ngramfile/',data={"hello":"world"})
#print ms.content

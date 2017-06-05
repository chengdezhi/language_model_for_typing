import sys
import csv
import time
import os

checked_num = 10000
def computeKSR(client,doc,sLocale="en"):
        cnt = 0
        sentences = doc.strip().split("\n")
        finalResult = []
        process_time = 0
        total_saved = 0  
        total_len = 0
        ksrfile = file('data/ksr_completion.csv','w')
        ksr = csv.writer(ksrfile)
        ksr.writerow(['text','process_time','predict_words','single_ksr'])
        for text in sentences:
            text = text.strip()
            if not text:
                continue
            if len(text.split(" "))<2 or len(text.split(" "))>=19:
                continue
            start = time.time()
            i =0
            predictWords = []
            total_len += len(text) 
            single_len = len(text)
            single_saved = 0
            while i < len(text):
                i += 1
                sWord = text[0:i]
                #sWord = sWord[max(len(sWord) - 50,0):]
                res = client.get(sWord)
                if sWord.endswith(' '): #is sWord end with space
                    #res = client.get(sWord)
                    nextSpaceIdx = text.find(' ' , i, len(text))
                    if nextSpaceIdx == -1 :
                        nextSpaceIdx = len(text)
                    word = text[i:nextSpaceIdx]
                    word = word.strip()
                    for result in res:
                        if word == result: #see the result is match   
                            #print( "ksr word -- matched ![%s]"% word)
                            predictWords.append(word)
                            forward = len(word)
                            i += forward
                            total_saved += len(word)
                            single_saved += len(word)
                            break
                else:
                    word,startIdx = findWord(text,i)
                    nextSpaceIdx = text.find(' ',i ,len(text))
                    #word_endIdx = nextSpaceIdx - 1
                    word = word.strip()
                    #import pdb
                    #pdb.set_trace()
                    for result in res:
                        if word == result:
                            predictWords.append(word[i-startIdx:])
                            print "predict:", word[i-startIdx:]
                            forward = len(word) - len(sWord[startIdx:i])
                            i += forward
                            total_saved += forward
                            single_saved += forward
                            break
            end  = time.time()
            process_time = end - start
            cnt += 1
            if cnt <= checked_num:
                #print(["ksr",process_time,text,predictWords])
                print("cnt:",cnt,"total ksr:",float(total_saved)/total_len,"saved:",total_saved,"total_len:",total_len)
                ksr.writerow([cnt, text, process_time, predictWords, float(single_saved)/single_len])
            if cnt == checked_num:
                print("total ksr:",float(total_saved)/total_len,"saved:",total_saved,"total_len:",total_len)
                ksr.writerow(["total ksr:",float(total_saved)/total_len,"saved:",total_saved,"total_len:",total_len])
                ksrfile.close()
        return total_saved*1.0/total_len

def findWord(text, i):
    start_idx = text.rfind(' ', 0 , i)
    end_idx = text.find(' ', i , len(text))
    if end_idx == -1:
        end_idx = len(text)
    word = text[ start_idx+1 : end_idx]
    return word, start_idx+1


if __name__ == "__main__":
    #print main("what a beautiful day\nas soon as possible")
    text = "what can i do for you"
    print(findWord(text,6))

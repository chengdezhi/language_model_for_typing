import sys
def computeKSR(client,doc,sLocale="en"):
        sentences = doc.strip().split("\n")
        finalResult = []
        for text in sentences:
            text = text.strip()
            if not text:
                continue
            i =0
            predictCount = 0
            while i < len(text):
                i += 1
                sWord = text[0:i]
                sWord = sWord[max(len(sWord) - 50,0):]
                #print("--you are typing:[%s]"% sWord)
                res = client.getPrediction(sWord, sLocale)
                predictCount += 1
                res = res.listWords
                if len(res)==0:
                    continue
                result = res[0]
                if not sWord.endswith(' '): #is sWord end with space
                    word,startIdx = findWord(text,i)
                    if word == result: #see the result is match   
                        #print( "matched !")
                        forward = len(word)-len(sWord[startIdx:i])
                        i += forward
                else:
                    nextSpaceIdx = text.find(' ' , i, len(text))
                    if nextSpaceIdx == -1 :
                        nextSpaceIdx = len(text)
                    word = text[i:nextSpaceIdx]
                    if word == result: #see the result is match   
                        #print( "--matched !")
                        forward = len(word)
                        i += forward
        #end while
            finalResult.append([text,len(text),predictCount,float("%.3f" % (float(predictCount) /len(text)))])
        return finalResult


def findWord(text, i):
    start_idx = text.rfind(' ', 0 , i)
    end_idx = text.find(' ', i , len(text))
    if end_idx == -1:
        end_idx = len(text)
    word = text[ start_idx+1 : end_idx]
    return word, start_idx+1


if __name__ == "__main__":
    print main("what a beautiful day\nas soon as possible")
    #text = "what can i do for you"
    #print(findWord(text,6))

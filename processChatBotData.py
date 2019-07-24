import time
def processData(originFile = './chatBotData/12万对话语料青云库.csv',
                outPutInFile = './chatBotData/question.txt',
                outPutOutFile = './chatBotData/answer.txt'):
    questionF = open(outPutInFile , 'w')
    answerF = open(outPutOutFile , 'w')
    with open(originFile,'r') as originF:
        for line in  originF.readlines():
            try:
                print(line)
                [question,answer] = line.strip().split('|')
                print(question.replace(' ',''))
                print(answer.replace(' ',''))
                if len(answer) < 29:
                    questionF.write(question.replace(' ','')+'\n')
                    answerF.write(answer.replace(' ','')+'\n')
            except:
                pass

    questionF.close()
    answerF.close()
    print("=======Finish process origin chatBot Data")

if __name__ == "__main__":
    processData()

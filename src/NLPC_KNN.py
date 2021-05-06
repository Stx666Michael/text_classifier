import numpy as np
from sklearn import neighbors


def getTextWords(path):
    f = open(path, "r", encoding = 'UTF-8')
    txt = f.read().lower()
    
    for ch in '!"#$%&()*+,-./;:<=>?@[\\]^‘_{|}~':
        txt = txt.replace(ch, " ")
    words = txt.split()
    
    f.close()
    return words


def getLineWords(path):
    f = open(path, "r", encoding = 'UTF-8')
    line = f.readline().lower()
    lines = []
    
    while line:
        for ch in '!"#$%&()*+,-./;:<=>?@[\\]^‘_{|}~':
            line = line.replace(ch, " ")
        words = line.split()
        
        lines.append(words)
        line = f.readline().lower()
        
    f.close()
    return lines


def getFreq(words):
    counts = {}
    
    for word in words:
        counts[word] = counts.get(word, 0) + 1
        
    items = list(counts.items())
    items.sort(key=lambda x:x[1], reverse=True)
    return items,counts


def printFreq(items):
    for i in range(len(items)):
        word, count = items[i]
        print("{0:<10}{1:>5}".format(word, count))


def readDataSet(path,numQues,numWords,lineFreq):
    f = open(path, "r")
    print("Sample:",numQues)
    dataSet = np.zeros([numQues,numWords],int)
    hwLabels = np.zeros([numQues])
    
    for i in range(numQues):   
        hwLabels[i] = f.readline()
        dataSet[i] = lineFreq[i]
        
    f.close()
    return dataSet,hwLabels


def getLineFreq(lines,counts):
    lineFreq = []
    
    for linewords in lines:
        lineItems,l_counts = getFreq(linewords)
        
        for word in counts.keys():
            for l_word in l_counts.keys():
                if l_word == word and l_word not in ['i','to','can','my','the','what','how','if']:
                    counts[word] = 1
                    break
                counts[word] = 0
                
        lineFreq.append(list(counts.values()))
        
    #print(lineFreq)
    return lineFreq


def getQuesFreq(counts,op):
    ques = op.lower()
    for ch in '!"#$%&()*+,-./;:<=>?@[\\]^‘_{|}~':
        ques = ques.replace(ch, " ")
    quesWords = ques.split()
    #print(quesWords)
    quesLines = []
    quesLines.append(quesWords)
    quesFreq = getLineFreq(quesLines,counts)
    #print(quesFreq)
    return quesFreq


def getQuesClass(res):
    QC = {}
    QC[0] = 'date'
    QC[1] = 'exchange, study abroad'
    QC[2] = 'course transfer, 2+2 & 4+0'
    QC[3] = 'health care'
    QC[4] = 'contact list'
    QC[5] = 'campus return'
    QC[6] = 'extenuating circumstances'
    QC[7] = 'feedback'
    QC[8] = 'teaching schedule'
    QC[9] = 'academic'
    return QC[res]
    

def main():
    words = getTextWords("hamlet_all.txt")
    items,counts = getFreq(words)
    #printFreq(items)
    #print(len(items))
    #print(counts)
    lines = getLineWords("hamlet_all.txt")
    lineFreq = getLineFreq(lines,counts)
        
    print("Training...")
    train_dataSet, train_hwLabels = readDataSet("labels_all.txt", len(lines), len(items), lineFreq)
    knn = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_neighbors=1)
    knn.fit(train_dataSet, train_hwLabels)
    print("Training complete.")
    
    print("\nTesting...")
    lines = getLineWords("hamlet_t.txt")
    lineFreq = getLineFreq(lines,counts)
    dataSet, hwLabels = readDataSet("labels_t.txt", len(lines), len(items), lineFreq)

    res = knn.predict(dataSet)
    error_num = np.sum(res != hwLabels)
    num = len(dataSet)
    print("Wrong:",error_num,"in",num,"\nAccuracy:",round(1-error_num/num,6))

    op = input("\nEnter your questions ('n' to quit): ")
    while op != 'n':
        res = knn.predict(getQuesFreq(counts,op))
        print(knn.kneighbors(getQuesFreq(counts,op)))
        print("Question Type:",getQuesClass(int(res)))        
        op = input("\nEnter your questions ('n' to quit): ")

main()

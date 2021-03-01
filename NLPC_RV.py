import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
import random


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
    print("Dataset:",numQues)
    dataSet = np.zeros([numQues,numWords],int)
    hwLabels = np.zeros([numQues,10])
    
    for i in range(numQues):
        digit = int(f.readline())
        hwLabels[i][digit] = 1.0
        dataSet[i] = lineFreq[i]
        
    f.close()
    return dataSet,hwLabels


def readDataSet_K(path,numQues,numWords,lineFreq):
    f = open(path, "r")
    hwLabels = np.zeros([numQues])
    
    for i in range(numQues):   
        hwLabels[i] = f.readline()
        
    f.close()
    return hwLabels


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


def getQuesClass(res,Res):
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
    for vec in res:
        for num in range(len(vec)):
            if vec[num] == 1:
                return QC[num],"(Classifier: MLP Neural Network)"
        return QC[Res],"(Classifier: KNN Algorithm)"


def getQues(path,lineNum):
    f = open(path, "r", encoding = 'UTF-8')
    for i in range(lineNum+1):
        line = f.readline()
    f.close()
    return line


def main():
    words = getTextWords("hamlet_all.txt")
    items,counts = getFreq(words)
    #printFreq(items)
    #print(len(items))
    #print(counts)
    lines = getLineWords("hamlet_all.txt")
    lineFreq = getLineFreq(lines,counts)
        
    trainNum = 635
    testTimes = 10
    print("Training...")
    train_dataSet, train_hwLabels = readDataSet("labels_all.txt", len(lines), len(items), lineFreq)
    print("Training Sample:",trainNum,"\nTesting Sample:",len(train_dataSet)-trainNum)

    clf = MLPClassifier(hidden_layer_sizes=(50,),
                        activation='logistic', solver='adam',
                        learning_rate_init = 0.001, max_iter=1000)
    print(clf,'\n')

    knn_hwLabels = readDataSet_K("labels_all.txt", len(lines), len(items), lineFreq)
    knn = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_neighbors=1)
    
    Knn = neighbors.NearestNeighbors(n_neighbors=3)
    Knn.fit(train_dataSet, knn_hwLabels)

    for j in range(testTimes):
        trainSample = []
        trainSampleLb = []
        trainSampleLb_K = []
        
        testSample = []
        testSampleLb = []
        testSampleLb_K = []

        Sample = random.sample(range(len(train_dataSet)),trainNum)

        for i in range(len(Sample)):
            trainSample.append(train_dataSet[Sample[i]])
            trainSampleLb.append(train_hwLabels[Sample[i]])
            trainSampleLb_K.append(knn_hwLabels[Sample[i]])

        for i in range(len(train_dataSet)):
            if i not in Sample:
                testSample.append(train_dataSet[i])
                testSampleLb.append(train_hwLabels[i])
                testSampleLb_K.append(knn_hwLabels[i])
        
        clf.fit(trainSample, trainSampleLb)
        knn.fit(trainSample, trainSampleLb_K)

        print("Random testing:",j+1,'/',testTimes)

        res = clf.predict(testSample)
        #print(res)
        error_num = 0
        num = len(testSample)
        for i in range(num):
            #比较长度为10的数组，返回包含01的数组，0为不同，1为相同
            #若预测结果与真实结果相同，则10个数字全为1，否则不全为1
            if np.sum(res[i] == testSampleLb[i]) < 10: 
                error_num += 1 
        print("Wrong:",error_num,"in",num,"\tAccuracy:",round(1-error_num/num,6),"(MLP Neural Network)")

        res = knn.predict(testSample)
        error_num = np.sum(res != testSampleLb_K)
        print("Wrong:",error_num,"in",num,"\tAccuracy:",round(1-error_num/num,6),"(KNN Algorithm)")
    

    print("\nTesting complete. Training with whole dataset...")
    clf.fit(train_dataSet, train_hwLabels)
    knn.fit(train_dataSet, knn_hwLabels)
    print("Training complete.")
    
    op = input("\nEnter your questions ('n' to quit): ")
    while op != 'n':
        a = getQuesFreq(counts,op)
        res = clf.predict(a)
        Res = knn.predict(a)
        for v in a:
            if sum(v) == 0:
                print("Unknown question type")
                break
            Class,Classifier = getQuesClass(res,int(Res))
            print("Question Type:",Class,Classifier)

            dist,neigh = Knn.kneighbors(a)
            #print(dist,neigh)
            for v in dist:
                maxDist = max(v)
            if maxDist >= 2:
                print("\nResults may compromise. Consider more questions:\n")
                for v in neigh:
                    for qNum in v:
                        print('\t',getQues("hamlet_all.txt",qNum),end='')
            
        op = input("\nEnter your questions ('n' to quit): ")

main()

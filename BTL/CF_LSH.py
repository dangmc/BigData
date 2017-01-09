import numpy as np
import math
import random
import csv

path = "/home/mliafol/Documents/HocTap/Big Data/BTL/Data/ml-l/"
number_user = 71567
number_item = 10681
#number_user = 943
#number_item = 1682
number_genre = 0
number_profile = 0
matrixContentUser = []
matrixContentItem = []
number_signature = 5000

fileTrainName = path + "ra.train";
fileTestName = path + "ra.test";


'''
def read_genre_item():
    fileGenreName = path + "u.genre";
    file_handle = open(fileGenreName, "r");
    number_genre = 0
    for line in file_handle:
        number_genre += 1
    file_handle.close()
    fileGenreName = path + "u.item";
    file_handle = open(fileGenreName, "r");
    itemGenre = []
    for k in xrange(number_item):
        itemGenre.append([])
    for line in file_handle:
        data = line.split('|')
        genre = []
        for k in xrange(5,len(data)):
            genre.append(int(data[k]))
        itemGenre[int(data[0]) - 1] = genre
    print itemGenre[0]
    file_handle.close()
    return number_genre, itemGenre

def read_profile_user():
    fileUser = path + "u.occupation"
    file_handle = open(fileUser, "r")
    number_profile = 3
    number_job = 0
    userJob = []
    for line in file_handle:
        number_job += 1
        userJob.append(line)
    file_handle.close()
    userProfile = []
    print userJob
    for k in xrange(number_user):
        userProfile.append([])
    fileUserProfile = path + "u.user"
    file_handle = open(fileUserProfile, "r")
    for line in file_handle:
        data = line.split("|")
        profile = []
        profile.append(int(data[1]))
        if (data[2] == "M"):
            profile.append(0)
        else:
            profile.append(1)
        cs = 0;
        for k in xrange(number_job):
            if (data[3] + "\n" == userJob[k]):
                cs = k;
                break;
        profile.append(cs)
        userProfile[int(data[0]) - 1] = profile
    print userProfile[0]
    return  number_profile, userProfile
'''

def readIdItem():
    fileName = path + "movies.dat"
    file_handle = open(fileName, "r")
    idItem = {}
    numberItem = 0
    for line in file_handle:
        data = line.split("::")
        item_id = int(data[0])
        idItem[item_id] = numberItem
        numberItem += 1
    #print idItem[65133]
    return idItem

def  readfile():
    idItem = readIdItem()
    user = []
    item = []
    userRating = []
    itemRating= []
    file_handle = open(fileTrainName, "r");
    for i in xrange(number_user):
        user.append([])
        userRating.append([])
    for i in xrange(number_item):
        item.append([])
        itemRating.append([])
    for line in file_handle:
        data = line.split('::')
        user_id = (int) (data[0])
        item_id = int (data[1])
        rate = float(data[2])
        itemId = idItem[item_id]
        #print "%d %d %f\n" %(user_id, itemId, rate)
        user[user_id - 1].append(itemId)
        item[itemId].append(user_id - 1)
        userRating[user_id - 1].append(rate)
        itemRating[itemId].append(rate)
    file_handle.close()
    return user, userRating, item, itemRating


def getCosin(v1, v2):
    length1 = 0
    length2 = 0
    for k in xrange(len(v1)):
        if (v1[k] == 1):
            length1 += 1
    for k in xrange(len(v2)):
        if (v2[k] == 1):
            length2 += 1
    result = 0
    length1 = math.sqrt(float(length1))
    length2 = math.sqrt(float(length2))
    for k in xrange(len(v1)):
        result += v1[k] * v2[k];
    return (float(result)/(length1 * length2))

user, ratingUser, item, ratingItem = readfile()
#number_genre, itemGenre = read_genre_item()
#number_profile, userProfile = read_profile_user()
lengthItem = []
averageRatingUser = []
averageRatingItem = []
estimateRatingUser = []
baseAverageRatingUser = []
baseAverageRatingItem = []
totalAverage = 0
numberAverage = 0
for k in xrange(number_user):
    average = 0;
    baseAverageRatingUser.append([])
    for i in xrange(len(user[k])):
        average += ratingUser[k][i]
        totalAverage += ratingUser[k][i]
    numberAverage += len(user[k])
    if (len(user[k]) != 0):
        average = float(average)/len(user[k])
    for i in xrange(len(user[k])):
        baseAverageRatingUser[k].append(ratingUser[k][i] - average)
    averageRatingUser.append(average)
totalAverage = totalAverage / numberAverage
for k in xrange(number_item):
    average = 0
    baseAverageRatingItem.append([])
    for i in xrange(len(item[k])):
        average += ratingItem[k][i]
    if (len(item[k]) != 0):
        average = float(average)/len(item[k])
    for i in xrange(len(item[k])):
        baseAverageRatingItem[k].append(ratingItem[k][i] - average)
    averageRatingItem.append(average)

for k in xrange(number_user):
    estimate = []
    for i in xrange(len(user[k])):
        bk = -totalAverage + averageRatingUser[k]
        bi = -totalAverage + averageRatingItem[user[k][i]]
        estimate.append(totalAverage + bk + bi)
    estimateRatingUser.append(estimate)
for i in xrange(number_item):
    lengthItem.append(0);
    for k in xrange(len(item[i])):
        lengthItem[i] += baseAverageRatingItem[i][k] *  baseAverageRatingItem[i][k];
    lengthItem[i] = math.sqrt((float)(lengthItem[i]));

def hashingArrayGenerate():
    hash_array = []
    for i in xrange(number_user):
        u = random.randint(1, 2);
        if (u == 1):
            hash_array.append(1)
        else:
            hash_array.append(-1)
    return hash_array

def min_hashing():
    signatureMatrixItem = []
    for i in xrange(number_signature):
        print "%d\n" %(i)
        hash_array = hashingArrayGenerate()
        signature = []
        for k in xrange(number_item):
            sum = 0
            for j in xrange(len(item[k])):
                sum += ratingItem[k][j] * hash_array[item[k][j]]
            if (sum > 0):
                signature.append(1)
            else:
                signature.append(0)
        signatureMatrixItem.append(signature)
    return signatureMatrixItem

def local_sensitive_hashing(signatureMatrixItem):
    band = 200
    r = number_signature/band
    giaithua = []
    giaithua.append(1)
    for k in xrange(1,r):
        giaithua.append(giaithua[k - 1] * 2)
    setSimilarItem = []
    valueSimilarItem = []
    for k in xrange(number_item):
        setSimilarItem.append(set())
        valueSimilarItem.append([])
    for k in xrange(band):
        print k
        firstRow = k * r
        bucket = []
        for i in xrange(1024):
            bucket.append([])
        for i in xrange(number_item):
            gt = 0
            for j in xrange(r):
                index = firstRow + j
                if (signatureMatrixItem[index][i] == 1):
                    gt = gt + giaithua[j]
            bucket[gt].append(i)
        for i in xrange(1024):
            for k in xrange(len(bucket[i])):
                for t in xrange(len(bucket[i])):
                    if (k != t):
                        setSimilarItem[bucket[i][k]].add(bucket[i][t])

    for k in xrange(number_item):
        setSimilarItem[k] = list(setSimilarItem[k])
        setSimilarItem[k] = sorted(setSimilarItem[k])
    for k in xrange(number_item):
        print "%d" %(k)
        for i in xrange(len(setSimilarItem[k])):
            valueSimilarItem[k].append(getSimilarity(k, setSimilarItem[k][i]))
    return setSimilarItem, valueSimilarItem

def getSimilarity(itemi, itemj):
    k = 0
    t = 0
    n = min(len(item[itemi]), len(item[itemj]))
    sum  = 0
    while (k < n and t < n):
        if (item[itemi][k] == item[itemj][t]):
            sum += ratingItem[itemi][k] * ratingItem[itemj][k];
            k += 1;
            t += 1;
        elif (item[itemi][k] < item[itemj][t]):
            k += 1;
        else:
            t += 1;
    if (lengthItem[itemi] != 0 and lengthItem[itemj] != 0):
        return sum/(lengthItem[itemi] * lengthItem[itemj])
    else:
        return 0;

def getSimilaritySign(itemi, itemj, signatureMatrix):
    res = 0
    for k in xrange(number_signature):
        if (signatureMatrix[k][itemi] != signatureMatrix[k][itemj]):
            res += 1
    return math.cos(math.pi * float(res)/number_signature)

'''
def contendBaseItemRecommend():
    for k in xrange(number_user):
        content = []
        numberOne = []
        for i in xrange(number_genre):
            content.append(0)
            numberOne.append(0)
        for j in xrange(len(user[k])):
            for i in xrange(number_genre):
                content[i] += ratingUser[j] * itemGenre[user[j]][i]
                if (itemGenre[user[j]][i] == 1):
                    numberOne[i] += 1
        for i in xrange(number_genre):
            if (numberOne[i] != 0):
                content[i] = float(content[i])/numberOne[i]
        matrixContentUser.append(content)
def contendBaseUserRecommend():
    for k in xrange(number_item):
        content = []
        numberOne = []
        for i in xrange(number_profile):
            content.append(0)
            numberOne.append(0)
        for j in xrange(len(item[k])):
            for i in xrange(number_genre):
                content[i] += ratingItem[j] * userProfile[item[j]][i]
            if (i[user[j]][i] == 1):
                numberOne[i] += 1
    for i in xrange(number_genre):
        content[i] = float(content[i]) / numberOne[i]
    matrixContentUser.append(content)
'''
numberSimilarItem = 100
def recommendError():
    idItem = readIdItem()
    error = 0
    numberTestElement = 0
    fileHandle = open(fileTestName,"r")
    for line in fileHandle:
        data  = line.split('::')
        user_id = (int)(data[0])
        item_id = int(data[1])
        user_id -= 1
        item_id =  idItem[item_id]
        rate = float(data[2])
        numberTestElement += 1
        sim = []
        predictRating = 0
        totalSim = 0
        for k in xrange(len(user[user_id])):
            sim.append(getSimilarity(user[user_id][k], item_id))
        for i in xrange(numberSimilarItem):
            maxSim = 0
            indexSim = -1
            for k in xrange(len(user[user_id])):
                if (maxSim < sim[k]):
                    maxSim = sim[k]
                    indexSim = k
            if (maxSim == 0):
                break
            predictRating += (ratingUser[user_id ][indexSim] - estimateRatingUser[user_id][indexSim])* maxSim;
            totalSim += maxSim;
            sim[indexSim] = 0;
        if (totalSim != 0):
             predictRating = (float)(predictRating) / totalSim
             buser = -totalAverage + averageRatingUser[user_id]
             bitem = -totalAverage + averageRatingItem[item_id]
             estimate = totalAverage + buser + bitem
             predictRating = estimate + predictRating
        else:
             predictRating = 3
        #print "%f %f %f %f %f" %(buser, bitem, estimate, predictRating, rate)
        print "%d %f %f" %(numberTestElement,rate, predictRating)
        error = error + (predictRating - rate) * (predictRating - rate)
    print "%d %f" %(numberTestElement, error)
    return math.sqrt(error / numberTestElement)
def recommendCFLSH():
    idItem = readIdItem()
    print "start minhashing\n"
    signatureMatrixItem = min_hashing()
    print "finish minhashing\n"
    setSimilarItem, valueSimilarItem= local_sensitive_hashing(signatureMatrixItem)
    print "finish lsh\n"
    error = 0
    numberTestElement = 0
    fileHandle = open(fileTestName, "r")
    for line in fileHandle:
        data = line.split('::')
        user_id = (int)(data[0])
        item_id = int(data[1])
        rate = float(data[2])
        item_id = idItem[item_id]
        user_id -= 1
        numberTestElement += 1
        sim1 = []
        sim2 = []
        indexSim = []
        predictRating = 0
        totalSim = 0
        k = 0;
        t = 0
        l1 = len(user[user_id])
        l2 = len(setSimilarItem[item_id])
        #print item_id
        #print setSimilarItem[item_id]
        while (k < l1 and t < l2):
            if (user[user_id][k] == setSimilarItem[item_id][t]):
                indexSim.append(k)
                #sim1.append(getSimilaritySign(user[user_id][k], item_id, signatureMatrixItem))
                sim2.append(valueSimilarItem[item_id][t])
                k += 1
                t += 1
            elif (user[user_id][k] < setSimilarItem[item_id][t]):
                k += 1
            else:
                t += 1
        #print sim2
        for k in xrange(len(indexSim)):
            if (sim2[k] >= 0.2):
                predictRating += (ratingUser[user_id][indexSim[k]] - estimateRatingUser[user_id][indexSim[k]]) * sim2[k];
                totalSim += sim2[k];
        if (totalSim != 0):
             predictRating = (float)(predictRating) / totalSim
             buser = -totalAverage + averageRatingUser[user_id]
             bitem = -totalAverage + averageRatingItem[item_id]
             estimate = totalAverage + buser + bitem
             predictRating = estimate + predictRating
        else:
             predictRating = 2.5
        #print "%d %f %f" %(numberTestElement,rate, predictRating)
        error = error + (predictRating - rate) * (predictRating - rate)
        print "%d %f %f" %(numberTestElement, predictRating, rate)
    print "%d %f" % (numberTestElement, error)
    return math.sqrt(error / numberTestElement)
    return error
#error = recommendError()
#print "Error = %f"  %error;
error = recommendCFLSH()
print "Error = %f"  %error


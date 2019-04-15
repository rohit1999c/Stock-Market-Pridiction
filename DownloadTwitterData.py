import get_twitter_data
import requests
import get_yahoo_data
import re
import tweepy
import csv
import sys
from StringIO import StringIO
from zipfile import ZipFile
from urllib import urlopen
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import Multiclass_SVM
from sklearn.naive_bayes import BernoulliNB
import NaiveBayes
from sklearn.metrics import confusion_matrix
import datetime
import ProcessCSV
# Get input from user
print "Enter any keyword listed below"
print "Example --> AAPL GOOG YHOO MSFT GS"
print "-------------------------------------------------"
print "-------------------------------------------------"

response = raw_input("Please enter Keyword: ")

while not response:
    response = raw_input("Please enter Keyword: ")

# Get Tweets
keyword = '$'+response
# time = 'today'
time = 'lastweek'

print "Fetch twitter data for "+ response+" company keyword...."

today=datetime.date.today()
twitterData = get_twitter_data.TwitterData(today.strftime("%Y-%m-%d"))
tweets = twitterData.getTwitterData(keyword, time)
print tweets
print "Twitter data fetched \n"

print "Fetch yahoo finance data for "+response+" given company keyword.... "

keyword2 = response
# yahooData = get_yahoo_data.YahooData('2019-03-07', "2019-03-06")
# historical_data = yahooData.getYahooData(keyword2)


ProcessCSV.getCSV(keyword2)                             #downloading CSV file from yahoo finance
historical_data_file=keyword2+".csv"
print "historical data is : ",historical_data_file

yahoo_open_price = {}  # declaring an empty python dictionary
yahoo_close_price = {}
yahoo_high_price = {}
yahoo_low_price = {}

#calculate number of rows in CSV file
'''
reader_file=csv.reader(historical_data_file);
rowcount=len(list(reader_file));
print "rowcount is :", rowcount
'''
fields = []

with open(historical_data_file,'r') as csvFile:
    csvreader = csv.reader(csvFile)
    fields = csvreader.next();  # to get the cursor in the second row.
    #print fields
    for row in csvreader:
        print row; # type of row is list.
        yahoo_open_price.update({row[0]: row[1]})
        yahoo_high_price.update({row[0]: row[2]})
        yahoo_low_price.update({row[0]: row[3]})
        yahoo_close_price.update({row[0]: row[4]})
    print "done\n"
    #print "open price is : ",yahoo_open_price
'''
for i in range(rowcount):
    date = historical_data[i]['Date'].replace("-","")
    yahoo_open_price.update({date: historical_data[i]['Open']})  # dictionary1.update(dictionary2); update the python dictionary;
    yahoo_close_price.update({date: historical_data[i]['Close']})
    yahoo_high_price.update({date: historical_data[i]['High']})
    yahoo_low_price.update({date: historical_data[i]['Low']})
print "Yahoo data fetched \n\n"
'''
print "Collect tweet and process twitter corpus...."
tweet_s = []
for key,val in tweets.items():
    for value in val:
        tweet_s.append(value)

csvFile = open('Data/SampleTweets.csv', 'w')
csvWriter = csv.writer(csvFile)


# start replaceTwoOrMore
def replaceTwoOrMore(s):
    # look for 2 or more repetitions of character
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
# end

# start process_tweet
def processTweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = tweet.strip('\'"')
    return tweet
# end

# start getStopWordList
def getStopWordList(stopWordListFileName):
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
# end

# start getfeatureVector
def getFeatureVector(tweet, stopWords):
    featureVector = []
    words = tweet.split()
    for w in words:
        w = replaceTwoOrMore(w)
        w = w.strip('\'"?,.')
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*$", w)
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector
# end

# start extract_features
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features
# end

def getFeatureVectorAndLabels(tweets, featureList):
    sortedFeatures = sorted(featureList)
    map = {}
    feature_vector = []
    labels = []
    file = open("newfile.txt", "w")

    for t in tweets:
        label = 0
        map = {}
        for w in sortedFeatures:
            map[w] = 0

        tweet_words = t[0]
        tweet_opinion = t[1]

        # Fill the map
        for word in tweet_words:
            word = replaceTwoOrMore(word)
            word = word.strip('\'"?,.')
            if word in map:
                map[word] = 1
        # end for loop
        values = map.values()
        feature_vector.append(values)
        if (tweet_opinion == '|positive|'):
            label = 0
            tweet_opinion = 'positive'
        elif (tweet_opinion == '|negative|'):
            label = 1
            tweet_opinion = 'negative'
        elif (tweet_opinion == '|neutral|'):
            label = 2
            tweet_opinion = 'neutral'
        labels.append(label)
        feature_vector_value = str(values).strip('[]')
        file.write(feature_vector_value + "," + str(label) + "\n")
    file.close()
    return {'feature_vector' : feature_vector, 'labels': labels}
#end

# Download the AFINN lexicon, unzip, and read the latest word list in AFINN-111.txt
url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
zipfile = ZipFile(StringIO(url.read()))
afinn_file = zipfile.open('AFINN/AFINN-111.txt')

afinn = dict()
for line in afinn_file:
    parts = line.strip().split()
    if len(parts) == 2:
        afinn[parts[0]] = int(parts[1])

def tokenize(text):
    return re.sub('\W+', ' ', text.lower()).split()

def afinn_sentiment(terms, afinn):

    total = 0.
    for t in terms:
        if t in afinn:
            total += afinn[t]
    return total

def sentiment_analyzer():
    tokens = [tokenize(t) for t in tweet_s]  # Tokenize all the tweets

    afinn_total = []
    for tweet in tokens:
        total = afinn_sentiment(tweet, afinn)
        afinn_total.append(total)

    positive_tweet_counter = []
    negative_tweet_counter = []
    neutral_tweet_counter = []
    for i in range(len(afinn_total)):
        if afinn_total[i] > 0:
            positive_tweet_counter.append(afinn_total[i])
            csvWriter.writerow(["|positive|", tweet_s[i].encode('utf-8').split("|")[0], tweet_s[i].encode('utf-8').split("|")[1], float(afinn_total[i])])
        elif afinn_total[i] < 0:
            negative_tweet_counter.append(afinn_total[i])
            csvWriter.writerow(["|negative|", tweet_s[i].encode('utf-8').split("|")[0], tweet_s[i].encode('utf-8').split("|")[1], float(afinn_total[i])])
        else:
            neutral_tweet_counter.append(afinn_total[i])
            csvWriter.writerow(["|neutral|", tweet_s[i].encode('utf-8').split("|")[0], tweet_s[i].encode('utf-8').split("|")[1], float(afinn_total[i])])

# Main

print "Processing tweets and store in CSV file ...."
sentiment_analyzer()

print "Tweet corpus processed \n "

print "Preparing dataset...."
# Read the tweets one by one and process it
inpTweets = csv.reader(open('Data/SampleTweets.csv', 'rb'), delimiter=',')
stopWords = getStopWordList('Data/stopwords.txt')
count = 0;
featureList = []
labelList = []
tweets = []
dates =[]
date_split =[]
list_tweet = []
print "Creating feature set and generating feature matrix...."
for row in inpTweets:
    if len(row) == 4:
        list_tweet.append(row)
        sentiment = row[0]
        date = row[1]
        t = row[2]

        date_split.append(date)
        dates.append(date)
        labelList.append(sentiment)
        processedTweet = processTweet(t)
        featureVector = getFeatureVector(processedTweet, stopWords)
        featureList.extend(featureVector)
        tweets.append((featureVector, sentiment));

result = getFeatureVectorAndLabels(tweets, featureList)

print "Dataset is ready \n"

print "Sentiment prediction using Naive Bayes Bernoulli and SVM model...."
# Naive Bernoulli and SVM Algorithm
data2 = open('newfile.txt', 'r')

inp_data2 = []
files = np.loadtxt(data2,dtype=str, delimiter=',')

inp_data2 = np.array(files[:,0:-1], dtype='float')
givenY = files[:,-1]

target2=np.zeros(len(givenY), dtype='int')
unique_y = np.unique(givenY)

for cls in range(len(givenY)):
    for x in range(len(unique_y)):
        if(givenY[cls] == unique_y[x]):
            target2[cls] = x

X = np.array(inp_data2)
y = np.array(target2)

# print type(X)
# print type(y)

max_gX = {}
maximum_gX = []
temp = 0
svn_temp = 0
final_precision=0
final_recall = 0
final_fmeasure = 0
final_accuracy = 0

svm_final_accuracy = 0
svm_final_precision = 0
svm_final_recall = 0
svm_final_fmeasure = 0

svm_accuracy = []
NB_accuracy = []
NBSKL_accuracy = []
print "X.shape is : ",X.shape[0];

kf = KFold(n_splits=6, shuffle=False)
print "get n splits :",kf.get_n_splits(X);
for train_index, test_index in kf.split(X):  # I(rohit) change it from kf to kf.split(X);
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # SVM Start
    print "SVM Start"
    clf = Multiclass_SVM.MulticlassSVM(C=0.1, tol=0.01, max_iter=100, random_state=0, verbose=1)
    clf.fit(X_train, y_train)
    predicted_y = clf.calculate_prediction(X_test)
    svm_accuracy.append(accuracy_score(y_test,predicted_y))
    svm_confusion_mat = confusion_matrix(y_test, predicted_y)
    sv_accuracy, svm_precision_val, svm_recall_val, svm_f_measure_val = clf.svm_findOtherParameters(svm_confusion_mat)

    # SVM end

    # Naive Bayes

    clf_NB = BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
    clf_NB.fit(X_train, y_train)
    predictedBNB_y = clf_NB.predict(X_test)
    NBSKL_accuracy.append(accuracy_score(y_test,predictedBNB_y))

    nb = NaiveBayes.NaiveBayesBernoulli()
    # iterate data for each class
    for clas in np.unique(y):
        class_feature_matrix = X_train[y_train==clas]
        prior_array = len(class_feature_matrix)*1.0/len(X_train)
        # print prior_array
        alpha = [(np.sum(class_feature_matrix[:,i])/len(class_feature_matrix)) for i in range(class_feature_matrix.shape[1])]
        gX = nb.membership_function(X_test, alpha, prior_array)
        max_gX.update({int(clas): gX})

    # find discriminant function
    disc_function = nb.discriminant_function(max_gX, np.unique(y))
    # print disc_function
    confusion_mat = confusion_matrix(y_test, predictedBNB_y)
    # print confusion_mat

    # find precision, recall , f-measure
    accuracy, precision_val, recall_val, f_measure_val = nb.findOtherParameters(confusion_mat)

    if accuracy_score(y_test,predictedBNB_y) > temp:
        if (accuracy_score(y_test,predictedBNB_y) != 1):
            final_accuracy = accuracy_score(y_test,predictedBNB_y)
            final_precision = precision_val
            final_recall = recall_val
            final_fmeasure = f_measure_val
            temp = accuracy_score(y_test,predictedBNB_y)

    if accuracy_score(y_test,predicted_y) > svn_temp:
        if (accuracy_score(y_test,predicted_y) != 1):
            svm_final_accuracy = accuracy_score(y_test,predicted_y)
            svm_final_precision = svm_precision_val
            svm_final_recall = svm_recall_val
            svm_final_fmeasure = svm_f_measure_val
            svn_temp = accuracy_score(y_test,predicted_y)

# Naive Bayes end
print "Bernoulli NB"
print "Accuracy =" ,max(NBSKL_accuracy)
print "Precision = ", final_precision
print "Recall = ", final_recall
print "F-Measure", final_fmeasure
print "\n"
print "SVM"
print "Accuracy =", max(svm_accuracy)
print "Precision = ", svm_final_precision
print "Recall = ", svm_final_recall
print "F-Measure", svm_final_fmeasure
print "\n"

print "Prediction completed \n"
print "Preparing dataset for stock prediction using yahoo finance and tweet sentiment...."

# making arrays for values for graph
date_totalCountArr = []
date_PosCountArr = []
date_NegCountArr = []
total_sentiment_scoreArr = []
date_openingprice = []
date_closingprice = []


date_tweet_details = {}
file = open("stockpredict.txt", "w")
date_arr = []
for dateVal in np.unique(date_split):
    date_totalCount = 0
    date_PosCount = 0
    date_NegCount = 0
    date_NutCount = 0
    total_sentiment_score = 0
    for row in list_tweet:
        sentiment = row[0]
        temp_date = row[1]
        sentiment_score = row[3]
        if(temp_date == dateVal):
            total_sentiment_score += int(float(sentiment_score))
            date_totalCount+=1
            if (sentiment == '|positive|'):
                date_PosCount+=1
            elif (sentiment == '|negative|'):
                date_NegCount+=1
            elif (sentiment == '|neutral|'):
                date_NutCount+=1

    s = str(date_totalCount)+" "+str(date_PosCount)+" "+str(date_NegCount)+" "+str(date_NutCount)
    date_tweet_details.update({dateVal: s})

    dateVal = dateVal.strip()
    print "Date value is : ",dateVal
    day = datetime.datetime.strptime(dateVal, '%Y-%m-%d').strftime('%A')
    closing_price = 0.
    opening_price = 0.
    if day == 'Saturday':
        '''update_date = dateVal.split("-")
        if len(str((int(update_date[2])+2)))==1:
            dateVal = update_date[0]+"-"+update_date[1]+"-0"+str((int(update_date[2])+2))
        else:
            dateVal = update_date[0] + "-" + update_date[1] + "-" + str((int(update_date[2]) + 2))
        opening_price = yahoo_open_price[dateVal]
        closing_price = yahoo_close_price[dateVal]
        '''
        print dateVal + " date check sat"
        continue;
    elif day == 'Sunday':
        ''' update_date = dateVal.split("-")
        if len(str((int(update_date[2])+1)))==1:
            dateVal = update_date[0]+"-"+update_date[1]+"-0"+str((int(update_date[2])+1))
        else:
            dateVal = update_date[0] + "-" + update_date[1] + "-" + str((int(update_date[2]) + 1))
        opening_price = yahoo_open_price[dateVal]
        closing_price = yahoo_close_price[dateVal]
        '''
        print dateVal + " date check sun"
        continue
    else:
        opening_price = yahoo_open_price[dateVal]
        closing_price = yahoo_close_price[dateVal]
        date_arr.append(dateVal)  # appending date one by one.
        print dateVal + " date check  normal day"

    print dateVal+" date check"   #output
    print "Total tweets = ", date_totalCount, " Positive tweets = ", date_PosCount, " Negative tweets = ", date_NegCount #output
    date_totalCountArr.append(date_totalCount)
    date_PosCountArr.append(date_PosCount)
    date_NegCountArr.append(date_NegCount)
    total_sentiment_scoreArr.append(total_sentiment_score)
    date_openingprice.append(opening_price)
    date_closingprice.append(closing_price)

    print "Total sentiment score = ", total_sentiment_score  #output
    print "Opening Price = ", opening_price        # output
    print "Closing Price = ", closing_price        #output

    market_status = 0
    if (float(closing_price)-float(opening_price)) > 0:
        market_status = 1
    else:
        market_status =-1
    file.write( str(date_PosCount) + "," + str(date_NegCount) + "," + str(date_NutCount) +"," + str(date_totalCount) + "," + str(market_status) + "\n")

    # print " Total Tweet For date =",dateVal ," Count =" , date_totalCount
    # print " Positive Tweet For date =",dateVal ," Count =" , date_PosCount
    # print " Negative Tweet For date =",dateVal ," Count =" , date_NegCount
    # print " Neutral Tweet For date =",dateVal ,"S Count =" , date_NutCount
file.close()

print "Dataset is ready for stock prediction \n"
# end




# not always working.
#code for 1st graph (date vs total sentiment score. and line for +ve, -ve and total tweets).
import matplotlib.pyplot as plt
# x axis values
#x = [1, 2, 3, 4, 5, 6]
# corresponding y axis values
# plotting the points
plt.plot(date_arr, date_totalCountArr, color='black', linestyle='dashed', linewidth=2, marker='.', markerfacecolor='black', markersize=12)
plt.plot(date_arr, date_PosCountArr, color='green', linestyle='dashed', linewidth=2, marker='.', markerfacecolor='green', markersize=12)
plt.plot(date_arr, date_NegCountArr, color='red', linestyle='dashed', linewidth=2, marker='.', markerfacecolor='red', markersize=12)
plt.bar(date_arr, total_sentiment_scoreArr, tick_label='', width=0.8, color=['grey'])

# setting x and y axis range
plt.ylim(1, 350)
plt.xlim(1, 7)

# naming the x axis
plt.xlabel('Date')
# naming the y axis
plt.ylabel('Sentiment Score')

# giving a title to my graph
plt.title('company data')


#putting value on each bar
import pandas as pd
freq_series = pd.Series.from_array( total_sentiment_scoreArr)
ax = freq_series.plot(kind='bar')

# For each bar: Place a label
for rect in ax.patches:
    # Get X and Y placement of label from rect.
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 2

    # Number of points between bar and label. Change to your liking.
    space = 5
    # Vertical alignment for positive values
    va = 'bottom'

    # If value of bar is negative: Place label below bar
    if y_value < 0:
        # Invert space to place label below
        space *= -1
        # Vertically align label at top
        va = 'top'

    # Use Y value as label and format number with one decimal place
    label = "{:.1f}".format(y_value)

    # Create annotation
    ax.annotate(
        label,  # Use `label` as label
        (x_value, y_value),  # Place label at end of the bar
        xytext=(0, space),  # Vertically shift label by `space`
        textcoords="offset points",  # Interpret `xytext` as offset in points
        ha='center')  # Horizontally center label
        #va=va)  # Vertically align label differently for
    # positive and negative values.


plt.xticks([0,1,2,3,4,5],date_arr)  #label to each bar on x axis

#save graph in directory
plt.savefig("company data image.png")

# function to show the plot
plt.show()





print "total tweets count : ",date_totalCountArr;
print "total -ve tweets count : ",date_NegCountArr;
print "total +ve tweets count : ",date_PosCountArr
print "total sentiment score : ",total_sentiment_scoreArr;
print "opening price array : ",date_openingprice;
print "closing price array : ",date_closingprice;
print "Date are : ",date_arr;
#import OutputGraph as OG
#OG.tweetsVstockprice(date_PosCountArr,date_NegCountArr,total_sentiment_score,date_arr,date_openingprice,date_closingprice);


import matplotlib.pyplot as plt1
import numpy as np1

#total_sentiment_score = [round(x) for x in  total_sentiment_score]
graphdata = np1.array([[0,0]]);
i = 0
while i < len(date_totalCountArr):
  temp1 = date_PosCountArr[i]
  temp2 = date_NegCountArr[i]
  temp = np1.array([[temp1,temp2]]);
  graphdata = np1.append(graphdata, temp, axis=0);
  i=i+1
print "graph data before delete : ",graphdata
graphdata = np1.delete(graphdata, (0), axis=0);
print "graph data is after delete : ",graphdata
length = len(graphdata);
# Set plot parameters
#fig, ax = plt1.subplots()
width = 0.5 # width of bar
x = np.arange(length)


print "graph data is : ",graphdata
plt1.bar(x, graphdata[:,0], width, color='#cc0000', label='Case-1',align='center',tick_label='')
plt1.bar(x + width, graphdata[:,1], width, color='#ff0066', label='Case-2',align='center',tick_label='')


# setting x and y axis range
plt1.ylim(1, 150)
plt1.xlim(1, 7)

# naming the x axis
plt1.xlabel('Date')
# naming the y axis
plt1.ylabel('No. of Tweets')

# giving a title to my graph
plt1.title('Correlation between tweet corpus and stock prices')
plt1.xticks([0,1,2,3,4,5],date_arr,rotation=90)
plt1.savefig("tweet corpus vs stock prices");
plt1.show()


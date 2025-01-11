import numpy 
import statistics
import re
import matplotlib.pyplot as plt
import math
import pandas as pd

#Q1
V1 = numpy.random.random(100)
V1_sorted = numpy.sort(V1)
print(V1_sorted)

#Q2
print(V1*3)

#Q3
print("mean of V1: ",numpy.mean(V1))
print("standard deviation of V1: ",statistics.stdev(V1))

#Q4
matrix=numpy.zeros((4,3))
print(matrix)
matrix=numpy.random.random((4,3))
print(matrix)

print("matrix in 1 dimensional form is: ",matrix.flatten())

#Q5
S1="I am a great listener. I am going to have an awesome life."
key="am"
index=S1.find(key)
print(key," found at index ",index)
occurences=S1.count(key)
print(key," found ",occurences," number of times")

#Q6
S2="I work hard and shall be rewarded"
S3=S1+S2
print(S3)

#Q7
words =re.split(r'[.\s]+', S3)
length1=len(words)
print(words)
print("length: ",length1)

#Q8
for i in words:
    if(i=='I'):
        words.remove(i)
    elif(i=="am"):
        words.remove(i)
    elif(i=="to"):
        words.remove(i)
    elif(i=="and"):
        words.remove(i)
    elif(len(i)>6):
        words.remove(i)
    else:
        continue
    
print(words)
length2=len(words)
print("length: ",length2)

#Q9
date_string = "01-JUN-2021"

day, month, year = date_string.split('-')

month_to_num = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
}

month_num = month_to_num[month.upper()]

print("Day: ",day)
print("Month: ",month)
print("Year: ",year)

#Q10
data = {
    "City": ["BENGALURU", "CHENNAI", "MUMBAI", "MYSURU", "PATNA", "JAMMU", "GANDHI NAGAR", "HYDERABAD", "ERNAKULAM", "AMARAVATI"],
    "State": ["KA", "TN", "MH", "KA", "BH", "JK", "GJ", "TS", "KL", "AP"],
    "PIN Code": [560001, 600001, 400001, 570001, 800001, 180001, 382001, 500001, 682001, 522001],
}

df = pd.DataFrame(data)
df.to_excel("q10.xlsx", index=False)

df_loaded = pd.read_excel("q10.xlsx")
df_loaded["City, State"] = df_loaded["City"] + ", " + df_loaded["State"]
df_loaded.to_excel("q10_updated.xlsx", index=False)

#Q11
x = numpy.arange(len(V1))
plt.plot(x, V1, marker='o', linestyle='-', color='red')
plt.show()

#Q12
V2=[]
for i in V1:
    V2.append(math.pow(i,2))

x = numpy.arange(len(V1))

plt.plot(x, V1, marker='o', linestyle='-', color='b', label='Array 1')
plt.plot(x, V2, marker='s', linestyle='--', color='r', label='Array 2')

plt.show()


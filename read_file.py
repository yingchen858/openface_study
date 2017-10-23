import pickle
import pandas as pd
import re
from numpy import linalg as LA
import numpy as np
from scipy.spatial.distance import pdist,squareform

def get_list(mydict, key, result):
   result=re.sub('\[', '', result)
   result=re.sub('\]', '', result)
   result=re.sub(' +',' ',result)
   result_list=result.strip().split(' ')
   mydict[key] = [float(x) for x in result_list]


f1=open('feature_extract_result.txt','r')

result =''
mydict = {}
for line in f1:
  if line[0]=='i':
     if result!='':
        get_list(mydict, key, result)
     key = int(''.join(x for x in re.findall(r'\d+', line)))
     result =''
  else :
     result += line.rstrip()

get_list(mydict, key, result)

#for key in mydict:
#   print key, LA.norm(np.array(mydict[key]))

DF_var = pd.DataFrame.from_dict(mydict).T
similarity = squareform(pdist(DF_var, metric='cosine'))

#print similarity[0][0]
#print DF_var

f2=open('mapping','r')
index_dict = {}
id =0 
for line in f2:
   id+=1
   l=line.strip().split(':')
   for n in range(int(l[0]),int(l[1])+1):
       index_dict[n] = id

truth = []
similarity_result = []

for i in range(1,len(mydict)+1):
   for j in range(i+1, len(mydict)+1):
       if index_dict[i] == index_dict[j]:
          truth.append(1)
       else:
           truth.append(0)
       similarity_result.append(similarity[i-1][j-1])

with open('truth.pkl', 'wb') as f:
   pickle.dump(truth, f)

with open('similarity_result.pkl', 'wb') as f:
   pickle.dump(similarity_result, f)

import pandas as pd 
from sklearn.model_selection import train_test_split

k = pd.read_csv('./SDP_train.csv')

print (k.head())
print (k.columns)

for i in k.columns :
    print (i)
    if i not in ['citing_title','cited_title','citation_context','citation_influence_label']:
        k.drop(i, inplace=True,axis =1 )


print (k.head())
print (k.columns)

X_train, Xtest, _, _ = train_test_split(k,[0]*3000 ,test_size=0.166666666)

print (X_train)
print (X_train.columns)

X_train.to_csv('train.csv',index=False,header=False)
Xtest.to_csv('validation.csv',index=False,header=False)

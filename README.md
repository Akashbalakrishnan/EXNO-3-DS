## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![Screenshot 2024-10-10 093600](https://github.com/user-attachments/assets/20000847-f17a-42ca-8c0e-741ffbb5859c)

## ORDINAL ENCODER
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![Screenshot 2024-10-10 093606](https://github.com/user-attachments/assets/a1d3a907-f161-4f08-9bc8-f5fec5a7640d)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![Screenshot 2024-10-10 093613](https://github.com/user-attachments/assets/9e406131-9446-4df5-94fa-8173727caf62)

## LABEL ENCODER
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(df[["ord_2"]])
dfc
```
![Screenshot 2024-10-10 093628](https://github.com/user-attachments/assets/c55b7f6c-b3be-42fe-be1c-8a40c3105d8a)

```
dfc=df.copy()
dfc['con_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![Screenshot 2024-10-10 093635](https://github.com/user-attachments/assets/d6aaacf1-e70c-4fd0-88f7-9f923c80bc4a)

## ONEHOT ENCODER
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df[['nom_0']]))
enc
```
![Screenshot 2024-10-10 093640](https://github.com/user-attachments/assets/1341ef40-7e7d-4168-ae45-f4963a901107)

```
df2=pd.concat([df,enc],axis=1)
df2
```
![Screenshot 2024-10-10 093647](https://github.com/user-attachments/assets/05b71cb4-a5ee-4879-8b17-b93f34251a77)

```
pip install --upgrade category_encoders
```
![Screenshot 2024-10-10 093716](https://github.com/user-attachments/assets/3faaa984-cc37-45f8-9bd2-0a4db9b59827)

## BinaryEncoder
```
from category_encoders import BinaryEncoder
import pandas as pd
df=pd.read_csv("/content/data (1).csv")
df
```
![Screenshot 2024-10-10 093728](https://github.com/user-attachments/assets/059925df-b78c-42e4-81d6-8924ae1534b9)

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![Screenshot 2024-10-10 093742](https://github.com/user-attachments/assets/2269a8ed-7228-485d-a45e-ef2e4fe0cb6c)

## TARGET ENCODER
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![Screenshot 2024-10-10 093752](https://github.com/user-attachments/assets/1f3a898d-c0c2-4240-b25b-46407134c8e9)

## FEATURE ENGINEERING
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![Screenshot 2024-10-10 093800](https://github.com/user-attachments/assets/baa476fe-b1ee-4023-9a85-14ade48de38f)

```
df.skew()
```
![Screenshot 2024-10-10 093808](https://github.com/user-attachments/assets/c937f082-9910-49a9-af8e-db8bb2fd5a28)

```
df["Highly Positive Skew"]=np.log(df["Highly Positive Skew"])
df
```
![Screenshot 2024-10-10 093820](https://github.com/user-attachments/assets/fd54cb53-a0f0-45dd-a77a-601e961588f6)

```
df["Moderate Positive Skew"]=np.reciprocal(df["Moderate Positive Skew"])
df
```
![Screenshot 2024-10-10 093830](https://github.com/user-attachments/assets/7d0fed2e-3258-4f50-a59a-ff6f675e2e60)

```
df["Highly Positive Skew"]=np.sqrt(df["Highly Positive Skew"])
df
```
![Screenshot 2024-10-10 093844](https://github.com/user-attachments/assets/c5012092-be56-41ea-88e4-aa69dc4634e9)

```
df["Highly Positive Skew"]=np.square(df["Highly Positive Skew"])
df
```
![Screenshot 2024-10-10 093855](https://github.com/user-attachments/assets/7a509ea9-8bb0-4481-a99a-7b9f9864d1b1)

## POWER TRANSFORMATION

```
df["Highly Positive Skew"],parameter=stats.boxcox(df["Highly Positive Skew"])
df
```
![Screenshot 2024-10-10 093910](https://github.com/user-attachments/assets/a313187e-408f-4cf8-87b8-551363595d11)

```
df["Moderate Negative Skew_yeojohnson"],parameter=stats.yeojohnson(df["Moderate Negative Skew"])
df
```
![Screenshot 2024-10-10 093924](https://github.com/user-attachments/assets/1e226708-28d1-4ea2-ba4a-52e71cd4c361)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2024-10-10 093934](https://github.com/user-attachments/assets/24b9158d-c6ee-4876-a9e0-97762b398474)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![Screenshot 2024-10-10 093948](https://github.com/user-attachments/assets/cc563934-093e-486d-8527-87ab4ce11a47)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2024-10-10 094005](https://github.com/user-attachments/assets/b379811a-5103-4923-b59e-009d29d0b099)

# RESULT:
    
Thus,the given data are read and Feature Encoding and Transformation process are performed and the data is saved to the file.

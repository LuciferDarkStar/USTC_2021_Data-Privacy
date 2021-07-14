import pandas as pd
import sys
import time

# 输入k
desired_k = sys.argv[1]
# 输入MaxSup
MaxSup = sys.argv[2]
desired_k = int(desired_k)
MaxSup = int(MaxSup)
start = time.time()
# 列名
names = (
    'age',
    'work_class',
    'final_weight',
    'education',
    'education_num',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital_gain',
    'capital_loss',
    'hours_per_week',
    'native_country',
    'class'
)

# 非数值型列
categorical = (
    'work_class',
    'education',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native_country',
    'class'
)

gender = (
    'Female',
    'Male'
)

Race = (
    'Other',
    'Amer-Indian-Eskimo',
    'Black',
    'White',
    'Asian-Pac-Islander'
)

NM = [
    'Never-married'
]

Married = [
    'Married-civ-spouse',
    'Married-AF-spouse'
]

leave = [
    'Divorced',
    'Separated'
]

alone = [
    'Widowed',
    'Married-spouse-absent'
]

# QI 和 S
QI_columns = ['age', 'sex', 'race', 'marital_status']
S_columns = ['occupation']

# 读数据
df = pd.read_csv("adult.data", sep=", ", header=None, names=names, index_col=False, engine='python')

# 将非数值列附上属性
for column in categorical:
    df[column] = df[column].astype('category')

# 删除包含？的行
for i in names:
    df = df[~df[i].isin(['?'])]


# 取高度为H的向量集合
def get_H(H):
    T = []
    for i in range(0, 5):  # age
        for j in range(0, 2):  # Gender
            for x in range(0, 2):  # Race
                for y in range(0, 3):  # marital_status
                    if i + j + x + y == H:
                        T.append([i, j, x, y])
    return T


# 关于年龄的划分
def Age(df, i, partition):
    rows = []
    temp = []
    if i == 4:  # *划分
        rows = partition
    elif i == 3:  # 区间20
        for j in partition:
            if len(j) > 1:
                f = df['age'][j]
                rows.append(f.index[f.between(71, 90)])
                rows.append(f.index[f.between(51, 70)])
                rows.append(f.index[f.between(31, 50)])
                rows.append(f.index[f.between(10, 30)])
            else:
                rows.append(j)
        for j in rows:  # 删除空的划分
            if len(j) > 0:
                temp.append(j)
        rows = temp
    elif i == 2:  # 区间10
        for j in partition:
            if len(j) > 1:
                f = df['age'][j]
                rows.append(f.index[f.between(81, 90)])
                rows.append(f.index[f.between(71, 80)])
                rows.append(f.index[f.between(61, 70)])
                rows.append(f.index[f.between(51, 60)])
                rows.append(f.index[f.between(41, 50)])
                rows.append(f.index[f.between(31, 40)])
                rows.append(f.index[f.between(21, 30)])
                rows.append(f.index[f.between(10, 20)])
            else:
                rows.append(j)
        for j in rows:  # 删除空的划分
            if len(j) > 0:
                temp.append(j)
        rows = temp
    elif i == 1:  # 区间5
        for j in partition:
            if len(j) > 1:
                f = df['age'][j]
                rows.append(f.index[f.between(86, 90)])
                rows.append(f.index[f.between(81, 85)])
                rows.append(f.index[f.between(76, 80)])
                rows.append(f.index[f.between(71, 75)])
                rows.append(f.index[f.between(66, 70)])
                rows.append(f.index[f.between(61, 65)])
                rows.append(f.index[f.between(56, 60)])
                rows.append(f.index[f.between(51, 55)])
                rows.append(f.index[f.between(46, 50)])
                rows.append(f.index[f.between(41, 45)])
                rows.append(f.index[f.between(36, 40)])
                rows.append(f.index[f.between(31, 35)])
                rows.append(f.index[f.between(26, 30)])
                rows.append(f.index[f.between(21, 25)])
                rows.append(f.index[f.between(16, 20)])
                rows.append(f.index[f.between(10, 15)])
            else:
                rows.append(j)
        for j in rows:  # 删除空的划分
            if len(j) > 0:
                temp.append(j)
        rows = temp
    else:  # 原始值
        for j in partition:
            if len(j) > 1:
                f = df['age'][j]
                if f.nunique() == 1:
                    rows.append(j)
                else:
                    for a in f.unique():
                        rows.append(f.index[f.isin([a])])
            else:
                rows.append(j)
    return rows


# 关于性别的划分
def Gender(df, i, partition):
    rows = []
    temp = []
    if i == 0:  # 划分
        for j in partition:
            if len(j) > 1:
                f = df['sex'][j]
                rows.append(f.index[f.isin(['Female'])])
                rows.append(f.index[f.isin(['Male'])])
            else:
                rows.append(j)
        for j in rows:  # 删除空的划分
            if len(j) > 0:
                temp.append(j)
        rows = temp
    else:  # *划分
        rows = partition
    return rows


# 关于种族的划分
def race(df, i, partition):
    rows = []
    temp = []
    if i == 0:  # 原始值划分
        for j in partition:
            if len(j) > 1:
                f = df['race'][j]
                rows.append(f.index[f.isin(['Other'])])
                rows.append(f.index[f.isin(['Amer-Indian-Eskimo'])])
                rows.append(f.index[f.isin(['Black'])])
                rows.append(f.index[f.isin(['White'])])
                rows.append(f.index[f.isin(['Asian-Pac-Islander'])])
            else:
                rows.append(j)
        for j in rows:  # 删除空的划分
            if len(j) > 0:
                temp.append(j)
        rows = temp
    else:  # *划分
        rows = partition
    return rows


# 关于婚姻状况的划分
def M_status(df, i, partition):
    rows = []
    temp = []
    if i == 2:  # *划分
        rows = partition
    elif i == 1:
        for j in partition:  # 一层泛化
            if len(j) > 1:
                m = df['marital_status'][j]
                rows.append(m.index[m.isin(NM)])
                rows.append(m.index[m.isin(Married)])
                rows.append(m.index[m.isin(leave)])
                rows.append(m.index[m.isin(alone)])
            else:
                rows.append(j)
        for j in rows:  # 删除空的划分
            if len(j) > 0:
                temp.append(j)
        rows = temp
    else:  # 原始值划分
        for j in partition:
            if len(j) > 1:
                m = df['marital_status'][j]
                if m.nunique() == 1:
                    rows.append(j)
                else:
                    for a in m.unique():
                        rows.append(m.index[m.isin([a])])
            else:
                rows.append(j)
        rows = partition
    return rows


# debug
"""
Vector = [[df.index]]
temp_vector = Vector.pop()
print(temp_vector)
temp = Gender(df, 1, temp_vector)
print(temp)
temp = Age(df, 4, temp)
print(temp)
T = get_H(4)
"""


# 判断在K条件下，删除的数据是否超过MaxSup，返回 True或False，和删除了小于k的划分后的新的划分
def is_K(partition, k, MaxSup):
    Sum = int(0)
    rows = []
    for i, row in enumerate(partition):
        l = int(len(row))
        if l < k:
            Sum = Sum + l
            if Sum > MaxSup:
                return False, rows
        else:
            rows.append(partition[i])
    return True, rows


# debug
"""
v = [[2, 4, 6, 7], [1], [8], [3]]
k = 2
t, v = is_K(v, 2, 2)
print(t)
print(v)
"""


# Samarati算法
def Samarati(df, desired_k, MaxSup):
    # 初始化
    low = 0
    H = 8
    sol = [df.index]
    sol_m = [4, 1, 1, 2]
    Vector = [[df.index]]
    Map = []
    while low < H:  # 开始迭代
        Try = int((low + H) / 2)
        T = get_H(Try)
        if len(Vector) > 0:
            temp_vector = Vector.pop()
        else:
            break
        for t in T:
            # 根据每个向量进行划分
            t1 = M_status(df, t[3], temp_vector)
            t1 = Age(df, t[0], t1)
            t1 = Gender(df, t[1], t1)
            t1 = race(df, t[2], t1)
            Map.append(t)
            # 压入栈
            Vector.append(t1)
        reach_k = False
        while Vector and (not reach_k):  # 进行判断
            vec = Vector.pop()
            m = Map.pop()
            satisfies, t2 = is_K(vec, desired_k, MaxSup)
            if satisfies:
                sol = t2
                sol_m = m
                reach_k = True
        if reach_k:
            H = Try
        else:
            low = Try + 1
    return sol, sol_m


# 输出处理过后的数据,并计算LM
def Final_Out(df, df_final, Map):
    rows = []
    i = Map[0]
    j = Map[1]
    x = Map[2]
    y = Map[3]
    LM1 = 0
    LM2 = 0
    LM3 = 0
    LM4 = 0
    num = 0
    for partition in df_final:
        for row in partition:
            line = ''
            # 输出处理后的年龄
            if i == 0:
                LM1 = LM1 + 0
                line = line + str(df['age'][row]) + ','
            elif i == 1:
                LM1 = LM1 + (5 - 1) / 79
                if int(df['age'][row]) >= 10 and int(df['age'][row]) <= 15:
                    line = line + '10~15' + ','
                elif int(df['age'][row]) <= 20:
                    line = line + '16~20' + ','
                elif int(df['age'][row]) <= 25:
                    line = line + '21~25' + ','
                elif int(df['age'][row]) <= 30:
                    line = line + '26~30' + ','
                elif int(df['age'][row]) <= 35:
                    line = line + '31~35' + ','
                elif int(df['age'][row]) <= 40:
                    line = line + '36~40' + ','
                elif int(df['age'][row]) <= 45:
                    line = line + '41~45' + ','
                elif int(df['age'][row]) <= 50:
                    line = line + '46~50' + ','
                elif int(df['age'][row]) <= 55:
                    line = line + '51~55' + ','
                elif int(df['age'][row]) <= 60:
                    line = line + '56~60' + ','
                elif int(df['age'][row]) <= 65:
                    line = line + '61~65' + ','
                elif int(df['age'][row]) <= 70:
                    line = line + '66~70' + ','
                elif int(df['age'][row]) <= 75:
                    line = line + '71~75' + ','
                elif int(df['age'][row]) <= 80:
                    line = line + '76~80' + ','
                elif int(df['age'][row]) <= 85:
                    line = line + '81~85' + ','
                else:
                    line = line + '86~90' + ','
            elif i == 2:
                LM1 = LM1 + (10 - 1) / 79
                if int(df['age'][row]) >= 10 and int(df['age'][row]) <= 20:
                    line = line + '10~20' + ','
                elif int(df['age'][row]) <= 30:
                    line = line + '21~30' + ','
                elif int(df['age'][row]) <= 40:
                    line = line + '31~40' + ','
                elif int(df['age'][row]) <= 50:
                    line = line + '41~50' + ','
                elif int(df['age'][row]) <= 60:
                    line = line + '51~60' + ','
                elif int(df['age'][row]) <= 70:
                    line = line + '61~70' + ','
                elif int(df['age'][row]) <= 80:
                    line = line + '71~80' + ','
                else:
                    line = line + '81~90' + ','
            elif i == 3:
                LM1 = LM1 + (20 - 1) / 79
                if int(df['age'][row]) >= 10 and int(df['age'][row]) <= 30:
                    line = line + '10~30' + ','
                elif int(df['age'][row]) <= 50:
                    line = line + '31~50' + ','
                elif int(df['age'][row]) <= 70:
                    line = line + '51~70' + ','
                else:
                    line = line + '71~90' + ','
            else:
                LM1 = LM1 + 1
                line = line + '*' + ','
            # 输出处理后的性别
            if j == 1:
                LM2 = LM2 + 1
                line = line + '*' + ','
            else:
                line = line + str(df['sex'][row]) + ','
            # 输出处理后的race
            if x == 1:
                LM3 = LM3 + 1
                line = line + '*' + ','
            else:
                line = line + str(df['race'][row]) + ','
            # 输出处理后的marital_status
            if y == 2:
                LM4 = LM4 + 1
                line = line + '*' + ','
            elif y == 1:
                if df['marital_status'][row] in NM:
                    line = line + 'NM' + ','
                elif df['marital_status'][row] in Married:
                    LM4 = LM4 + 1 / 6
                    line = line + 'Married' + ','
                elif df['marital_status'][row] in leave:
                    LM4 = LM4 + 1 / 6
                    line = line + 'leave' + ','
                else:
                    LM4 = LM4 + 1 / 6
                    line = line + 'alone' + ','
            else:
                line = line + str(df['marital_status'][row]) + ','
            # 输出S集
            line = line + str(df['occupation'][row])
            rows.append(line)
        num = num + len(partition)
    LM = (LM1 + LM2 + LM3 + LM4) / num
    return rows, LM


df_final, Map = Samarati(df, desired_k, MaxSup)
print(Map)
Final, LM = Final_Out(df, df_final, Map)
print("LM=%f" % LM)
end = time.time()
print('Running time: %s Seconds' % (end - start))
with open('Final.txt', 'w') as file:
    for line in Final:
        file.write(str(line) + "\n")

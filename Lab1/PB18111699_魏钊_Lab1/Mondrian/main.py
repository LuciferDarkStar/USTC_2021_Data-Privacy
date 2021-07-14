# coding=utf-8
import pandas as pd
import sys
import time

# 输入k
desired_k = sys.argv[1]
desired_k = int(desired_k)
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

# QI 和 S
QI_columns = ['age', 'education_num']
S_columns = ['occupation']

# 读数据
df = pd.read_csv("adult.data", sep=", ", header=None, names=names, index_col=False, engine='python')

# 将非数值列附上属性
for column in categorical:
    df[column] = df[column].astype('category')

# 删除包含？的行
for i in names:
    df = df[~df[i].isin(['?'])]

# 总数
Sum = len(df)
# print(Sum)

# age总跨度
Age_span = df['age'].max() - df['age'].min()
# print(Age_span)

# education_num总跨度
Edu_span = df['education_num'].max() - df['education_num'].min()


# 对需要处理的每列进行跨度计算
def Column_span(df, partition, scale=None):
    spans = {}
    for column in df.columns:
        # 非数值型处理
        if column in categorical:
            span = len(df[column][partition].unique())
        else:
            # 数值型处理
            span = df[column][partition].max() - df[column][partition].min()
        if scale is not None:
            span = span / scale[column]
        spans[column] = span
    return spans


# 计算数据初始的跨度,这里计算了所有的列的跨度
full_spans = Column_span(df, df.index)


# 这里分别处理了数值型和非数值型的分割
def divide(df, partition, column):
    dfp = df[column][partition]
    if column in categorical:  # 非数值型
        values = dfp.unique()
        lv = set(values[:len(values) // 2])
        rv = set(values[len(values) // 2:])
        return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]
    else:  # 数值型
        median = dfp.median()
        dfl = dfp.index[dfp < median]
        dfr = dfp.index[dfp >= median]
        return (dfl, dfr)


# 判断分割后的项目数是否还是>=k
def Less_K(df, partition, sensitive_column, k=desired_k):
    if len(partition) < k:
        return False
    return True


# Mondrian算法
def Mondrian(df, QI_columns, S_columns, scale, Less_K):
    final_partitions = []
    partitions = [df.index]
    while partitions:
        partition = partitions.pop(0)
        spans = Column_span(df[QI_columns], partition, scale)
        # 循环根据相关列进行分割
        for column, span in sorted(spans.items(), key=lambda x: -x[1]):
            lp, rp = divide(df, partition, column)
            if not Less_K(df, lp, S_columns) or not Less_K(df, rp, S_columns):
                continue
            partitions.extend((lp, rp))
            break
        else:
            final_partitions.append(partition)
    return final_partitions


# 初始化
start_spans = Column_span(df, df.index)
# 计算最终划分
final_partitions = Mondrian(df, QI_columns, S_columns, full_spans, Less_K)

# debug
"""
for partition in final_partitions:
	print(len(partition))

print(final_partitions)
"""


# 结果输出，及LM计算
def Final_Out(df, partitions, feature_columns, sensitive_column, LM):
    rows = []
    for partition in partitions:
        for j in partition:
            line = ' '
            for column in feature_columns:
                if column in categorical:
                    diff_cate = df[column][partition].unique()
                    for k in diff_cate:
                        line = line + str(k) + '\000' + 'or' + '\000'
                    line = line[:-3]
                    line = line + ','
                else:
                    M = df[column][partition].max()
                    m = df[column][partition].min()
                    if column == 'age':
                        LM = LM + ((M - m) / Age_span) / Sum
                    if column == 'education_num':
                        LM = LM + ((M - m) / Edu_span) / Sum
                    if M != m:
                        line = line + str(m) + '~' + str(M) + ','
                    else:
                        line = line + str(M) + ','
            for column in sensitive_column:
                line = line + str(df[column][j]) + ','
            line = line[:-1]
            rows.append(line)
    return rows, LM


# 初始化
LM = 0

df_final, LM = Final_Out(df, final_partitions, QI_columns, S_columns, LM)
# 写入文件
print("LM=%f" % LM)
end = time.time()
print('Running time: %s Seconds' % (end - start))
with open('Final.txt', 'w') as file:
    for line in df_final:
        file.write(str(line) + "\n")

from scipy.stats import stats, f_oneway

def perform_anova(df, target):
    df_without_target = df.drop(target, axis = 1)
    headers = df_without_target.columns.values

    for i, header in enumerate(headers):
        print(i, " ", header)

    first_group = headers[int(input("Please enter the index of first group of ANOVA:  "))]
    second_group = headers[int(input("Please enter the index of second group of ANOVA:  "))]

    df_1 = df[[target, first_group]]
    df_2 = df[[target, second_group]]
    f_val, p_val = stats.f_oneway(df_1, df_2)

    print("F value: ", f_val[-1], "P value: ", p_val[-1])
    if (p_val[-1] < 0.05):
        print("There is a strong correlation between ", first_group," ", second_group)
    else:
        print("There is a week correlation between ", first_group," ", second_group)
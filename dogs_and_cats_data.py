import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as st
from sklearn.preprocessing import OrdinalEncoder


def get_age(age):
    age = age.split(" ")
    time = age[1]
    num = age[0]
    if time == "year" or time == "years":
        return float(num) * 12
    if time == "month" or time == "months":
        return float(num)
    if time == "week" or time == "weeks":
        return round(float(num) / 4)
    if time == "day" or time == "days":
        return round(float(num) / (4 * 7))


def simplify_age(age):
    if age < 6:
        return "0-6"
    if age < 12:
        return "6-12"
    if age < 36:
        return "12-36"
    if age < 62:
        return "36-62"
    if age < 96:
        return "62-96"
    else:
        return "96-.."

def get_name_length(name):
    if pd.isnull(name):
        return -1
    return len(name)

def has_name(name):
    if pd.isnull(name):
        return "Named"
    return "Not Named"


def simplify_color(color):
    return color.split("/")[0].split(" ")[0]


def is_mix(breed):
    if pd.isnull(breed):
        return "Mix"
    if breed.find("/") != -1 or breed.endswith(' mix'):
        return "Mix"
    return "Not Mix"


def is_spayed_neutered(sex):
    if pd.isnull(sex):
        return "Unknown"
    if sex.startswith("Spayed") or sex.startswith("Neutered"):
        return "Spayed"
    return "Intact"


def is_male(sex):
    if pd.isnull(sex):
        return "Unknown"
    if sex.endswith("Male"):
        return "Male"
    elif sex.endswith("Female"):
        return "Female"
    else:
        return "Unknown"


def simplify_breed(breed):
    breed = breed.split("/")[0]
    if breed.endswith(' Mix') or breed.endswith(' Mix'):
        breed = breed[:-4]

    # if value_count[breed] < 100:
    #     return "None"
    return breed


def simplify_breed2(breed, count):
    if count[breed] < 300:
        return "None"
    return breed


def encode_dates(df, hot_encode):
    dates = pd.to_datetime(df.DateTime)
    year = dates.dt.year
    year = year - year.min()
    year.name = "Year"
    month = dates.dt.month
    month.name = "Month"
    day = dates.dt.day
    day.name = "Day"
    hour = dates.dt.hour
    hour.name = "Hour"

    weekday = dates.dt.dayofweek
    weekday.name = "Weekday"
    if hot_encode:
        weekday = pd.get_dummies(weekday, prefix="Weekday")

    return hour, year, month, day, weekday


def encode_breed_color(df, hot_encode):
    ord_enc = OrdinalEncoder()
    if hot_encode:
        breed = pd.get_dummies(df['Breed'])
        color = pd.get_dummies(df['Color'])
        animal_data = pd.concat([breed, color], axis=1)
    else:
        animal_data = pd.DataFrame(ord_enc.fit_transform(df[['Breed', 'Color']]),
                                   columns=['Breed', 'Color'])

    return animal_data


def disp_stat(df, counter):
    train = df.copy()
    ages = train.AgeuponOutcome[train.AgeuponOutcome.notna()]
    ages = ages.apply(get_age)
    mean_age = ages[train.AgeuponOutcome.notna()].mean()
    train["AgeuponOutcome"][train.AgeuponOutcome.isnull()] = str(int(mean_age)) + " months"
    train["AgeuponOutcome"] = train["AgeuponOutcome"].apply(get_age)

    train['Color'] = train.Color.apply(simplify_color)
    value_counts = train.Breed.apply(simplify_breed).value_counts()
    train['Breed'] = train.Breed.apply(simplify_breed).apply(simplify_breed2, args=(value_counts, ))
    columns = ['AnimalType', 'SexuponOutcome', 'AgeuponOutcome', 'Breed', 'Color']
    for column in columns:
        if column == 'AgeuponOutcome':
            train[column].value_counts().sort_values(ascending=False).sort_index().plot(kind='bar', stacked=False,
                                                                                        figsize=(10, 8), rot=-45)
            plt.xlabel("Months")
        else:
            train[column].value_counts().sort_values(ascending=False).plot(kind='bar', stacked=False,
                                                                           figsize=(10, 8), rot=-45)

        plt.title(column)
        path = '/home/user348/Downloads/ml_project/stats/' + str(counter)
        plt.savefig(path)
        counter = counter + 1
        print(counter)


def disp(df, type, counter):
    train = df.copy()
    ages = train.AgeuponOutcome[train.AgeuponOutcome.notna()]
    ages = ages.apply(get_age)
    mean_age = ages[train.AgeuponOutcome.notna()].mean()
    train["AgeuponOutcome"][train.AgeuponOutcome.isnull()] = str(int(mean_age)) + " months"
    train["AgeuponOutcome"] = train["AgeuponOutcome"].apply(get_age)

    train['SexuponOutcome'][train['SexuponOutcome'].isnull()] = 'Unknown'

    train['Sex'] = train.SexuponOutcome.apply(is_male)
    train['Neutered'] = train.SexuponOutcome.apply(is_spayed_neutered)
    train['Age'] = train.AgeuponOutcome.apply(simplify_age)
    train['Name Length'] = train.Name.apply(get_name_length)
    train['Named'] = train.Name.apply(has_name)
    train['Mix'] = train.Name.apply(is_mix)
    train['hour'], train['year'], train['month'], train['day'], train['weekday'] = encode_dates(train, False)
    train['Color'] = train.Color.apply(simplify_color)
    train['Breed'] = train.Breed.apply(simplify_breed)
    counts = train.Breed.value_counts()
    train['Breed'] = train.Breed.apply(simplify_breed2, args=(counts, ))

    columns = ['Neutered', 'Sex', 'Age', 'Named', 'Name Length', 'Mix', 'hour', 'year', 'month', 'day',
               'weekday', 'Color', 'Breed']
    for column in columns:
        if type == "All":
            train[['OutcomeType', column, 'AnimalType']].groupby(['AnimalType', 'OutcomeType', column]).size() \
                .groupby(level=0).apply(lambda x: 100 * x / x.sum()).unstack().unstack() \
                .plot(kind='bar', stacked=False, figsize=(10, 8), rot=-30)
        else:
            train[['OutcomeType', column]].groupby([column, 'OutcomeType']).size().groupby(level=0) \
                .apply(lambda x: 100 * x / x.sum()).unstack().plot(kind='bar', rot=-30, stacked=True, figsize=(10,8))

        plt.title(type)
        path = '/home/user348/Downloads/ml_project/figs/' + str(counter)
        plt.savefig(path)
        counter = counter + 3
        print(counter)


train = pd.read_csv("train.csv")
train_dogs = train[train["AnimalType"] == "Dog"]
train_cats = train[train["AnimalType"] == "Cat"]

counter = 1
disp(train, "All", counter)
counter = 2
disp(train_dogs, "Dog", counter)
counter = 3
disp(train_cats, "Cat", counter)


counter = 1
disp_stat(train, counter)
# plt.show()






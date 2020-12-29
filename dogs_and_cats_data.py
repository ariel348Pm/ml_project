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
        return 6
    if age < 12:
        return 12
    if age < 36:
        return 36
    if age < 62:
        return 62
    if age < 96:
        return 96
    else:
        return 100


def hair_length(breed):
    if breed.find("Shorthair") != -1:
        return 1
    if breed.find("Longhair") != -1:
        return 2
    return 1


def aggresive(breed):
    if breed.find("Pit Bull") != -1:
        return 6
    if breed.find("Rottweiler") != -1:
        return 4
    if breed.find("Shepherd") != -1:
        return 2
    return 0


def get_name_length(name):
    if pd.isnull(name):
        return -1
    return len(name)


def has_name(name):
    if pd.isnull(name):
        return "Not Named"
    return "Named"


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


def shared_dates(date, dates):
    if len(dates[dates == date]) > 1:
        return 1
    return 0


def breed_pop(breed, breed_popularity):
    for i, p in enumerate(breed_popularity["BREED"]):
        if p.find(breed) != -1:
            if breed_popularity["2019"][i] < 25:
                return 2
            else:
                return 1
    return 0


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


def name_pop(name, count):
    if pd.isnull(name):
        return -1
    if count[name] > 4:
        return 2
    return 1


def colors(color):
    if color.find("Tricolor") != -1:
        return 3
    if color.find("/") != -1:
        return 2
    return 1


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
    minute = dates.dt.minute
    minute.name = "Minute"

    sum_minutes = minute + 60 * hour
    sum_minutes.name = "Sum_Minute"

    sum_days = day + 365 * year
    sum_days.name = "Sum_Days"

    weekday = dates.dt.dayofweek
    weekday.name = "Weekday"
    if hot_encode:
        weekday = pd.get_dummies(weekday, prefix="Weekday")

    return hour, year, month, day, weekday, minute, sum_minutes, sum_days


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
    train["AgeuponOutcome"] = train["AgeuponOutcome"].apply(get_age).apply(simplify_age)
    ages_list = ["0-6" + "[m]", "6-12" + "[m]", "12-36" + "[m]", "36-62" + "[m]", "62-96" + "[m]", "96-.." + "[m]"]

    train['Color'] = train.Color.apply(simplify_color)
    value_counts = train.Breed.apply(simplify_breed).value_counts()
    train['Breed'] = train.Breed.apply(simplify_breed).apply(simplify_breed2, args=(value_counts, ))
    columns = ['AnimalType', 'SexuponOutcome', 'AgeuponOutcome', 'Breed', 'Color']
    for column in columns:
        if column == 'AgeuponOutcome':
            train[column].value_counts().sort_index().plot(kind='bar', stacked=False,
                                                                                        figsize=(10, 8), rot=-45)
            plt.xticks(range(len(ages_list)), ages_list)
        else:
            train[column].value_counts().sort_values(ascending=False).plot(kind='bar', stacked=False,
                                                                           figsize=(10, 8), rot=-45)

        plt.title(column)
        path = './stats/' + str(counter)
        plt.savefig(path)
        counter = counter + 1
        print(counter)


def disp(df, type, counter, norm, path):
    train = df.copy()
    ages = train.AgeuponOutcome[train.AgeuponOutcome.notna()]
    ages = ages.apply(get_age)
    ages_list = ["0-6" + "[m]", "6-12" + "[m]", "12-36" + "[m]", "36-62" + "[m]", "62-96" + "[m]", "96-.." + "[m]"]

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
    train['hour'], train['year'], train['month'], train['day'], train['weekday'], train['minute'], train['sum_minutes']\
        , train['sum_days'] = encode_dates(train, False)
    weekdays = ['Mon', 'Tues', 'Weds', 'Thurs', 'Fri', 'Sat', 'Sun']
    breed_popularity = pd.read_csv("breeds.csv", sep="\t")

    train["Aggressiveness"] = train['Breed'].apply(aggresive)
    train["Hair"] = train['Breed'].apply(hair_length)
    train["numColors"] = train["Color"].apply(colors)
    train['Color'] = train.Color.apply(simplify_color)
    train['Breed'] = train.Breed.apply(simplify_breed)

    counts = train.Name.value_counts()
    train['NamePopularity'] = train.Name.apply(name_pop, args=(counts, ))

    train["SharedDates"] = train.groupby("DateTime")["DateTime"].transform('count')

    train["BreedPopularity"] = train['Breed'].apply(breed_pop, args=(breed_popularity,))

    columns = ['Neutered', 'Sex', 'Age', 'Named', 'Name Length', 'Mix', 'hour', 'year', 'month', 'day',
               'weekday', 'Color', 'Breed', 'Hair', 'Aggressiveness', 'BreedPopularity', 'NamePopularity',
               'numColors', 'SharedDates']
    for column in columns:
        if type == "All":
            train[['OutcomeType', column, 'AnimalType']].groupby(['AnimalType', 'OutcomeType', column]).size() \
                .groupby(level=0).apply(lambda x: 100 * x / x.sum()).unstack().unstack() \
                .plot(kind='bar', stacked=False, figsize=(10, 8), rot=-30)
        else:
            if norm:
                train[['OutcomeType', column]].groupby([column, 'OutcomeType']).size().groupby(level=0) \
                    .apply(lambda x: 100 * x / x.sum()).unstack().plot(kind='bar', rot=-30, stacked=True, figsize=(10,8))
            else:
                train[['OutcomeType', column]].groupby([column, 'OutcomeType']).size()\
                    .unstack().plot(kind='bar', rot=-30, stacked=True, figsize=(10, 8))

        if column == "weekday":
            plt.xticks(range(len(weekdays)), weekdays)
        if column == "Age":
            plt.xticks(range(len(ages_list)), ages_list)

        # if column == "Age":
        #     plt.xticks(range(len(weekdays)), weekdays)

        plt.title(type)
        plt.xlabel(column)

        num = str(counter)
        if len(num) == 1:
            num = "00" + num
        if len(num) == 2:
            num = "0" + num
        path1 = path + num + column
        path2 = './figs/' + num + column
        plt.savefig(path1)
        plt.savefig(path2)
        counter = counter + 7
        print(counter)


train = pd.read_csv("train.csv")
train_dogs = train[train["AnimalType"] == "Dog"]
train_cats = train[train["AnimalType"] == "Cat"]

counter = 1
path = "./all types/"
disp(train, "All", counter, False, path)
plt.close('all')
counter = 2
path = "./dogs/"
disp(train_dogs, "Dog", counter, False, path)
plt.close('all')
counter = 3
path = "./cats/"
disp(train_cats, "Cat", counter, False, path)
plt.close('all')
counter = 4
path = "./dogs/"
disp(train_dogs, "Dog", counter, True, path)
plt.close('all')
counter = 5
path = "./cats/"
disp(train_cats, "Cat", counter, True, path)
plt.close('all')
counter = 6
path = "./all/"
disp(train, "all", counter, False, path)
plt.close('all')
counter = 7
path = "./all/"
disp(train, "all", counter, True, path)
plt.close('all')


counter = 1
disp_stat(train, counter)
# plt.show()







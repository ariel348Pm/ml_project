import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt




def load_data():
    train_org = pd.read_csv("train.csv")
    test_org = pd.read_csv("test.csv")

    train = train_org.drop(['OutcomeSubtype', "AnimalID"], axis='columns', inplace=False)
    test = test_org.drop(["ID"], axis='columns', inplace=False)
    return train, test


def extract_data_features(train, test, animal_type):
    test_type = test[test['AnimalType'] == animal_type].drop(["AnimalType"], axis='columns', inplace=False)
    train_type = train[train['AnimalType'] == animal_type].drop(["AnimalType"], axis='columns', inplace=False)
    data_type = train_type.append(test_type, ignore_index=True)
    return data_type


def get_age(age):
    age = age.split(" ")
    time = age[1]
    num = age[0]
    if time == "year" or time == "years":
        return float(num) * 12
    if time == "month" or time == "months":
        return float(num)
    if time == "week" or time == "weeks":
        return float(num) / 4
    if time == "day" or time == "days":
        return float(num) / (4 * 7)

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

def get_name_length(name):
    return len(name)


def simplify_color(color):
    return color.split("/")[0].split(" ")[0]


def is_mix(breed):
    if breed.endswith(' Mix') or breed.endswith(' Mix'):
        return 1
    if breed.find("/") != -1:
        return 1
    return 0


def is_spayed_neutered(sex):
    if sex.startswith("Spayed") or sex.startswith("Neutered"):
        return 1
    return 0

def is_male(sex):
    if sex.endswith("Male"):
        return 1
    elif sex.endswith("Female"):
        return -1
    else:
        return 0


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


def simplify_breed(breed):
    breed = breed.split("/")[0]
    if breed.endswith(' Mix'):
        breed = breed[:-4]
    # for b in ["Sheepdog", "Pit Bull", "Terrier", "Schnauzer", "Retriever", "Shepherd", "Chihuahua", "Collie"]:
    #     if breed.find(b) != -1:
    #         return b
    return breed


def name_pop(name, count):
    if pd.isnull(name):
        return -1
    if count[name] > 4:
        2
    return 1


def breed_pop(breed, breed_popularity):
    for i, p in enumerate(breed_popularity["BREED"]):
        if p.find(breed) != -1:
            if breed_popularity["2019"][i] < 25:
                return 2
            else:
                return 1
    return 0


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

    weekday = dates.dt.dayofweek
    weekday.name = "Weekday"
    if hot_encode:
        weekday = pd.get_dummies(weekday, prefix="Weekday")

    date_data = pd.concat([hour, year, month, day, weekday], axis=1)

    return date_data


def encode_breed_color(df):
    ord_enc = OrdinalEncoder()
    if hot_encode:
        breed = pd.get_dummies(df['Breed'])
        color = pd.get_dummies(df['Color'])
        animal_data = pd.concat([breed, color], axis=1)
    else:
        animal_data = pd.DataFrame(ord_enc.fit_transform(df[['Breed', 'Color']]),
                                   columns=['Breed', 'Color'])

    return animal_data


def encode_features(df_in, hot_encode):
    df = df_in.copy()
    # Encode has name
    hasname = df.Name.isnull().astype(int)
    # Encode name
    counts = df.Name.value_counts()
    name_popularity = df.Name.apply(name_pop, args=(counts,))
    name_popularity.name = "Name Popularity"

    names_length = df.Name
    names_length[df.Name.notna()] = df.Name[df.Name.notna()].apply(get_name_length)
    names_length[df.Name.isnull()] = -1
    names_length.name = "NamesLength"

    # Encode date
    date_data = encode_dates(df, hot_encode)

    # Encode age
    ages = df.AgeuponOutcome
    ages[df.AgeuponOutcome.notna()] = df.AgeuponOutcome[df.AgeuponOutcome.notna()].apply(get_age)
    mean_age = ages[df.AgeuponOutcome.notna()].mean()
    ages[df.AgeuponOutcome.isnull()] = mean_age
    ages.name = "Age"
    # ages = ages.apply(simplify_age)

    num_colors = df["Color"].apply(colors)
    num_colors.name = "Num Colors"
    # Encode sex, breed and color
    df['SexuponOutcome'][df['SexuponOutcome'].isnull()] = 'Unknown'
    df['Color'] = df['Color'].apply(simplify_color)
    df['Breed'] = df['Breed'].apply(simplify_breed)

    animal_data = encode_breed_color(df)

    breed_popularity = df['Breed'].apply(breed_pop, args=(pd.read_csv("breeds.csv", sep="\t"),))
    breed_popularity.name = "Breed Popularity"

    mix = df['Breed'].copy().apply(is_mix)
    spayed = df['SexuponOutcome'].apply(is_spayed_neutered)
    spayed.name = "Spayed"
    male = df['SexuponOutcome'].apply(is_male)
    male.name = "Sex"
    aggressiveness = df['Breed'].apply(aggresive)
    aggressiveness.name = "Aggressiveness"
    hair = df['Breed'].apply(hair_length)
    hair.name = "Hair"

    shared_dates = df.groupby("DateTime")["DateTime"].transform('count')
    shared_dates.name = "Shared dates"

    columns = ['numColors']

    res = pd.concat([hasname, date_data, ages, animal_data, spayed, male, mix, aggressiveness,
                     breed_popularity, name_popularity, shared_dates, num_colors], axis=1)
    # return res.drop(["Breed", "Color", "NamesLength", "Year", "Day"], axis=1)
    return res


def encodeLabels(df):
    return df['OutcomeType']


def encode_test_train(data, encode, size):
    features = encode_features(data, encode)
    X_train = features[:size]
    X_test = features[size:].reset_index().drop(["index"], axis='columns', inplace=False)
    y = encodeLabels(data[:size])
    return X_train, X_test, y


def getModel(X, y, hot_encode, specifications, type):
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # # Feature Scaling
    if not hot_encode:
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

    # clf = RandomForestClassifier(n_estimators=250)
    # clf = AdaBoostClassifier(n_estimators=specifications["n_estimators"],
    #                          learning_rate=specifications["learning_rate"], random_state=0)
    clf = GradientBoostingClassifier(n_estimators=specifications["n_estimators"],
                                     learning_rate=specifications["learning_rate"],
                                     max_depth=specifications["max_depth"], random_state=0)
    # clf = KNeighborsClassifier(n_neighbors=7)


    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)

    prediction = pd.DataFrame(y_pred)
    prediction.columns = ['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']

    print(type + " Log-loss:", metrics.log_loss(y_test, y_pred))
    return clf, metrics.log_loss(y_test, y_pred), len(X_test)


def pred(clf, test_features):
    prediction = pd.DataFrame(clf.predict_proba(test_features))
    prediction.columns = ['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']
    return prediction


def get_id(test, animals):
    test_id = pd.Series([i+1 for i in range(len(test))], name="ID")
    animals_id = pd.Series(dtype="float64")
    for animal_type in animals:
        type_id = test_id[test["AnimalType"] == animal_type]
        if animals_id.empty:
            animals_id = type_id
        else:
            animals_id = pd.concat([animals_id, type_id], axis=0, ignore_index=True)

    return animals_id


def final_score(loss_dogs, loss_cats, num_dogs, num_cats):
    loss = (loss_dogs * num_dogs + loss_cats * num_cats) / (num_dogs + num_cats)
    print("Final Log-loss:" + str(loss))


def write_results(animals_id, prediction, path):
    prediction_all_types = pd.concat([animals_id, pd.concat([prediction["Cat"], prediction["Dog"]], axis=0,
                                                            ignore_index=True)], axis=1)
    prediction_all_types = prediction_all_types.sort_values(by=['ID'])
    prediction_all_types.to_csv(path, index=False)


hot_encode = False
animals = ["Cat", "Dog"]
prediction = {}

clf_specification = {}
clf_specification_dogs = {"n_estimators": 250, "learning_rate": 0.2, "max_depth": 2}
clf_specification_cats = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}
clf_specification["Dog"] = clf_specification_dogs
clf_specification["Cat"] = clf_specification_cats

train, test = load_data()
loss = {}
num_animal = {}
for animal_type in animals:
    size = len(train[train["AnimalType"] == animal_type])
    data = extract_data_features(train, test, animal_type)
    train_features, test_features, train_labels = encode_test_train(data, hot_encode, size)
    clf, loss[animal_type], num_animal[animal_type] = getModel(train_features, train_labels, hot_encode,
                                                               clf_specification[animal_type], animal_type)
    prediction[animal_type] = pred(clf, test_features)

    importance = clf.feature_importances_
    plt.figure(figsize=(20, 10))
    sns.barplot(y=train_features.columns, x=importance, ci=None)
    plt.savefig(animal_type)

final_score(loss["Dog"], loss["Cat"], num_animal["Dog"], num_animal["Cat"])
animals_id = get_id(test, animals)
path = "results.csv"
write_results(animals_id, prediction, path)


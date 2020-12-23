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


def simplify_breed(breed):
    breed = breed.split("/")[0]
    if breed.endswith(' Mix'):
        breed = breed[:-4]
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
    # Encode name length
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

    # Encode sex, breed and color
    df['SexuponOutcome'][df['SexuponOutcome'].isnull()] = 'Unknown'
    df['Color'] = df['Color'].apply(simplify_color)
    df['Breed'] = df['Breed'].apply(simplify_breed)

    animal_data = encode_breed_color(df)

    mix = df['Breed'].copy().apply(is_mix)
    spayed = df['SexuponOutcome'].apply(is_spayed_neutered)
    male = df['SexuponOutcome'].apply(is_male)

    res = pd.concat([hasname, names_length, date_data, ages, animal_data, spayed, male, mix], axis=1)
    return res.drop(["Breed", "Color", "NamesLength", "Year", "Day"], axis=1)


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

final_score(loss["Dog"], loss["Cat"], num_animal["Dog"], num_animal["Cat"])
animals_id = get_id(test, animals)
path = "results.csv"
write_results(animals_id, prediction, path)

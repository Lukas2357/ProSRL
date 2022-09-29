import pandas as pd

from helper import load_data
from helper import save_data


def generate_data_tests():
    df = load_data('../../data/data_trafo.csv')
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df.drop(df[df['Category'] == -1].index, inplace=True)
    df.drop(df[df['Label'] == 'Tour'].index, inplace=True)
    df.drop(df[df['Label'] == 'Startseite'].index, inplace=True)
    df.drop(df[df['Label'] == 'Übersicht'].index, inplace=True)
    df.drop(df[df['Label'] == 'Rückmeldungen'][df['TestResQual'] == -1].index, inplace=True)
    df.reset_index(inplace=True)
    df_user = df.groupby('User')
    df_mikro = []
    for user in df_user:
        index_tests = user[1].loc[user[1]['TestResQual'] >= 0].index

        for index in index_tests:
            df_mikro.append([user[0],
                             df.iloc[index - 1]['Category'],
                             df.iloc[index]['Category'],
                             df.iloc[index]['TestResQual'],
                             df.iloc[index + 1]['Category']])

    df_mikro = pd.DataFrame(df_mikro, columns=['User', 'PrevAct', 'Test', 'TestResQual', 'NextAct'])

    save_data(df_mikro, 'data_mikro_tests')


def generate_data_exercise():
    raise NotImplementedError()


def load_micro_tests():
    df = load_data('data_mikro_tests.csv')
    df = df[['User', 'PrevAct', 'Test', 'TestResQual', 'NextAct']][1:]
    return df


def load_micro_exercise():
    raise NotImplementedError()
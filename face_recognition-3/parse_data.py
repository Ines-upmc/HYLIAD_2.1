import pandas as pd
#import textwrap

def parse_csv(fname):
    df = pd.read_csv(fname)
    c = list(df.columns)

    name = list(df['Name'])
    birthday = list(df['Birthday'])
    age = list(df['Age'])
    profession = list(df['Profession'])
    mac = list(df['MAC'])
    height = list(df['Height'])
    weight = list(df['Weight'])
    eye = list(df['Eye_abbreviation'])
    sex = list(df['Sex'])
    ethnicity = list(df['Country_abbreviated'])
    file_name = list(df['File_name'])
    biography = list(df['Record'])
    biography = ["BUREAU RECORD- "+biography[i] for i in range(len(biography))]
    '''
    def fasta(txt):
        des = textwrap.wrap(txt, 55)
        des_ = '\n'.join(des)
        return des_
    
    biography = [fasta(biography[i]) for i in range(len(biography))] 
    '''
    return name, birthday, age, profession, mac, height, weight, eye, sex, ethnicity, file_name, biography

#parse_csv("database.csv")

import random

def parse_database_logs(fname):
    df = pd.read_csv(fname)
    names = list(df["Name"])
    d_ = range(1,30)
    d_ = ["0"+str(d_[i]) for i in range(9)]+[str(d_[i]) for i in range(9,len(d_))]
    dates = [d_[i]+"/04/2022" for i in range(len(d_))]

    def rand_d():
        h_entree = random.randint(10,16)
        m_entree = random.randint(10,59)
        t_entree = str(h_entree)+':'+str(m_entree)
        h_sorti = random.randint(16,21)
        m_sorti = random.randint(10,59)
        t_sorti = str(h_sorti)+':'+str(m_sorti)
        return t_entree, t_sorti
    
    def generate_daily(date):
        c= [date.replace('/','')+'_entree',date.replace('/','')+'_sortie']
        t_e = [rand_d()[0] for i in range(len(df))]
        t_s = [rand_d()[1] for i in range(len(df))]
        #dfa = pd.DataFrame(t_e)
        #dfa.columns = [c[0]]
        #dfa[c[1]] = t_s
        t_d = [t_e,t_s]
        tup_a = (c,t_d)
        return tup_a

    x = [generate_daily(dates[i]) for i in range(len(dates))]
    cs = [x[i][0] for i in range(len(x))]
    cs = sum(cs,[])
    dt = [x[i][1] for i in range(len(x))]
    dt = sum(dt,[])
    dfa = pd.DataFrame(dt)
    dfa.index = cs
    dfb = dfa.T
    dfb['Name'] = names
    dff = df.merge(dfb, how='left', on='Name')
    dff.to_csv("database_logs_parsed.csv")
    return

parse_database_logs("database_students.csv")

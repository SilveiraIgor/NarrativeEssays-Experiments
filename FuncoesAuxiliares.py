import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn import tree
from xgboost import XGBRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectKBest, f_regression

def get_dados(comp):
    pairs = {'C1': 'formal_register', 'C2': 'thematic_coherence', 'C3': 'narrative_rhetorical_structure', 'C4': 'cohesion'}
    OBJECTIVE = 'classification'
    enem = f'C{comp}'
    nome_bert = f"Bert-{OBJECTIVE}-{pairs[enem]}--Narrativo.csv"
    path_atual = os.getcwd()
    path_dados = os.path.join(path_atual, "dataset")
    train = pd.read_csv(os.path.join(path_dados, "traincleanNarrativoFeatures.csv"), encoding='utf-8')
    validation = pd.read_csv(os.path.join(path_dados, "validationcleanNarrativoFeatures.csv"), encoding='utf-8')
    test = pd.read_csv(os.path.join(path_dados, "testcleanNarrativoFeatures.csv"), encoding='utf-8')
    
    train_bert_classification = pd.read_csv(os.path.join(path_dados, f"train{nome_bert}"), encoding='utf-8')
    validation_bert_classification = pd.read_csv(os.path.join(path_dados, f"validation{nome_bert}"), encoding='utf-8')
    test_bert_classification = pd.read_csv(os.path.join(path_dados, f"test{nome_bert}"), encoding='utf-8')
    
    dic = {}
    dic['treinamento'] = train
    dic['validacao'] = validation
    dic['teste'] = test
    dic['treinamento_bert_classification'] = train_bert_classification
    dic['validacao_bert_classification'] = validation_bert_classification
    dic['teste_bert_classification'] = test_bert_classification
    
    OBJECTIVE = 'regression'
    nome_bert = f"Bert-{OBJECTIVE}-{pairs[enem]}--Narrativo.csv"
    train_bert = pd.read_csv(os.path.join(path_dados, f"train{nome_bert}"), encoding='utf-8')
    validation_bert = pd.read_csv(os.path.join(path_dados, f"validation{nome_bert}"), encoding='utf-8')
    test_bert = pd.read_csv(os.path.join(path_dados, f"test{nome_bert}"), encoding='utf-8')
    dic['treinamento_bert_regression'] = train_bert
    dic['validacao_bert_regression'] = validation_bert
    dic['teste_bert_regression'] = test_bert
    
    nome_base = "essays-pt-br-feats-"
    train_pteu = pd.read_csv(os.path.join(path_dados, f"{nome_base}train.csv"), encoding='utf-8')
    validation_pteu = pd.read_csv(os.path.join(path_dados, f"{nome_base}validation.csv"), encoding='utf-8')
    test_pteu = pd.read_csv(os.path.join(path_dados, f"{nome_base}test.csv"), encoding='utf-8')
    
    alinhamento_treinamento = pd.read_csv(os.path.join(path_dados, f"alinhamento_treinamento.csv"), encoding='utf-8')
    alinhamento_validacao = pd.read_csv(os.path.join(path_dados, f"alinhamento_validacao.csv"), encoding='utf-8')
    alinhamento_teste = pd.read_csv(os.path.join(path_dados, f"alinhamento_teste.csv"), encoding='utf-8')
    dic['treinamento_pteu'] = pd.merge(train_pteu, alinhamento_treinamento, on='id', how='inner')
    dic['validacao_pteu'] = pd.merge(validation_pteu, alinhamento_validacao, on='id', how='inner')
    dic['teste_pteu'] = pd.merge(test_pteu, alinhamento_teste, on='id', how='inner')
    return dic

def selecionar_comp(comp):
    return lambda row: eval(row['grade'])[comp-1]

def get_dados_tratados(comp):
    dados = get_dados(comp)
    dados['treinamento']['competencia'] = dados['treinamento'].apply(selecionar_comp(comp), axis=1)
    dados['treinamento'] = dados['treinamento'].drop(['grade', 'index_text'], axis=1)
    dados['teste']['competencia'] = dados['teste'].apply(selecionar_comp(comp), axis=1)
    dados['teste'] = dados['teste'].drop(['grade', 'index_text'], axis=1)
    dados['validacao']['competencia'] = dados['validacao'].apply(selecionar_comp(comp), axis=1)
    dados['validacao'] = dados['validacao'].drop(['grade', 'index_text'], axis=1)
    return dados

def filtrar_colunas(lista_colunas):
    filtradas = []
    terminacoes = ("_sum", "_min", "_max", "_median", "_q1", "_q3", "_80p", "_90p", "_mode", "_var", "_std", "_rsd", "_iqr", "_dolch", "_skewness", "_kurtosis")
    for f in lista_colunas:
        if f.endswith(terminacoes):
            filtradas.append(f)
    return filtradas

def arrumar_dados_treinamento(comp, tipo):
    dados = get_dados_tratados(comp)
    if tipo == 'NILC_merge':
        treinamento = pd.concat([dados['treinamento'], dados['validacao']] , axis=0   )
    elif tipo == 'bert_classification':
        dados = get_dados(comp)
        dados['treinamento']['competencia'] = dados['treinamento'].apply(selecionar_comp(comp), axis=1)
        dados['treinamento'] = pd.merge(dados['treinamento'], dados['treinamento_bert_classification'], on='index_text', how='inner').drop(['grade', 'index_text', 'confidence_0', 'confidence_1', 'confidence_2', 'confidence_3', 'confidence_4', 'confidence_5'], axis=1)
        dados['teste']['competencia'] = dados['teste'].apply(selecionar_comp(comp), axis=1)
        dados['teste'] = pd.merge(dados['teste'], dados['teste_bert_classification'], on='index_text', how='inner').drop(['grade', 'index_text', 'confidence_0', 'confidence_1', 'confidence_2', 'confidence_3', 'confidence_4', 'confidence_5'], axis=1)
        dados['validacao']['competencia'] = dados['validacao'].apply(selecionar_comp(comp), axis=1)
        dados['validacao'] = pd.merge(dados['validacao'], dados['validacao_bert_classification'], on='index_text', how='inner').drop(['grade', 'index_text', 'confidence_0', 'confidence_1', 'confidence_2', 'confidence_3', 'confidence_4', 'confidence_5'], axis=1)
        treinamento = pd.concat([dados['treinamento'], dados['validacao']] , axis=0   )
    elif tipo == 'bert_regression':
        dados = get_dados(comp)
        dados['treinamento']['competencia'] = dados['treinamento'].apply(selecionar_comp(comp), axis=1)
        dados['treinamento'] = pd.merge(dados['treinamento'], dados['treinamento_bert_regression'], on='index_text', how='inner').drop(['grade', 'index_text', ], axis=1)
        dados['validacao']['competencia'] = dados['validacao'].apply(selecionar_comp(comp), axis=1)
        dados['validacao'] = pd.merge(dados['validacao'], dados['validacao_bert_regression'], on='index_text', how='inner').drop(['grade', 'index_text'], axis=1)
        treinamento = pd.concat([dados['treinamento'], dados['validacao']] , axis=0   )
    elif tipo in ['neurais_merge', 'neurais']:
        dados = get_dados(comp)
        colunas = dados['treinamento'].columns
        dados['treinamento']['competencia'] = dados['treinamento'].apply(selecionar_comp(comp), axis=1)
        dados['treinamento'] = pd.merge(dados['treinamento'], dados['treinamento_bert_regression'], on='index_text', how='inner')
        dados['treinamento'] = pd.merge(dados['treinamento'], dados['treinamento_bert_classification'], on='index_text', how='inner')
        dados['treinamento'] = dados['treinamento'].drop(colunas, axis=1)
        
        dados['validacao']['competencia'] = dados['validacao'].apply(selecionar_comp(comp), axis=1)
        dados['validacao'] = pd.merge(dados['validacao'], dados['validacao_bert_regression'], on='index_text', how='inner')
        dados['validacao'] = pd.merge(dados['validacao'], dados['validacao_bert_classification'], on='index_text', how='inner')
        dados['validacao'] = dados['validacao'].drop(colunas, axis=1)
        if tipo == 'neurais_merge':
            treinamento = pd.concat([dados['treinamento'], dados['validacao']] , axis=0   )
        else:
            treinamento = dados['treinamento']
    elif tipo in ['iR4S', 'iR4S_merge', 'pteu_full', 'pteu_full_sem_merge', 'iR4S-F', 'iR4S-F_merge']:
        dados = get_dados(comp)
        colunas = list(dados['treinamento'].columns)
        colunas.remove('index_text')
        colunas.remove('grade')
        colunas.remove('pronoun_ratio')
        colunas.remove('ttr')
        if tipo in ['iR4S',  'iR4S-F']:
            dados['treinamento']['competencia'] = dados['treinamento'].apply(selecionar_comp(comp), axis=1)
            dados['treinamento'] = pd.merge(dados['treinamento'], dados['treinamento_pteu'],on='index_text', how='inner').drop(['id', 'grade', 'index_text', 'ttr_x', 'pronoun_ratio_x']+colunas, axis=1)
            treinamento = dados['treinamento']
            if tipo == 'iR4S-F':
                lista_colunas = treinamento.columns
                filtrar = filtrar_colunas(lista_colunas)
                treinamento = treinamento.drop(filtrar, axis=1)
        elif tipo in ['iR4S_merge',  'iR4S-F_merge']:
            dados['treinamento']['competencia'] = dados['treinamento'].apply(selecionar_comp(comp), axis=1)
            dados['treinamento'] = pd.merge(dados['treinamento'], dados['treinamento_pteu'],on='index_text', how='inner').drop(['id', 'grade', 'index_text', 'ttr_x', 'pronoun_ratio_x']+colunas, axis=1)
            dados['validacao']['competencia'] = dados['validacao'].apply(selecionar_comp(comp), axis=1)
            dados['validacao'] = pd.merge(dados['validacao'], dados['validacao_pteu'],on='index_text', how='inner').drop(['id', 'grade', 'index_text', 'ttr_x', 'pronoun_ratio_x']+colunas, axis=1)
            treinamento = pd.concat([dados['treinamento'], dados['validacao']], axis=0)
            if tipo == 'iR4S-F_merge':
                lista_colunas = treinamento.columns
                filtrar = filtrar_colunas(lista_colunas)
                treinamento = treinamento.drop(filtrar, axis=1)
        elif tipo in ['pteu_full', 'pteu_full_sem_merge']:
            dados['treinamento']['competencia'] = dados['treinamento'].apply(selecionar_comp(comp), axis=1)
            dados['treinamento'] = pd.merge(dados['treinamento'], dados['treinamento_pteu'],on='index_text', how='inner')
            dados['treinamento'] = pd.merge(pd.merge(dados['treinamento'], dados['treinamento_bert_regression'], on='index_text', how='inner'), dados['treinamento_bert_classification'], on='index_text', how='inner').drop(['grade', 'index_text', 'id'], axis=1)
            dados['validacao']['competencia'] = dados['validacao'].apply(selecionar_comp(comp), axis=1)
            dados['validacao'] = pd.merge(dados['validacao'], dados['validacao_pteu'],on='index_text', how='inner')
            dados['validacao'] = pd.merge(pd.merge(dados['validacao'], dados['validacao_bert_regression'], on='index_text', how='inner'), dados['validacao_bert_classification'], on='index_text', how='inner').drop(['grade', 'index_text', 'id'], axis=1)
            if tipo in ['pteu_full']:
                treinamento = pd.concat([dados['treinamento'], dados['validacao']], axis=0)
            else:
                treinamento = dados['treinamento']
    elif tipo in ['nao_neurais', 'nao_neurais_merge']:
        dados = get_dados(comp)
        dados['treinamento']['competencia'] = dados['treinamento'].apply(selecionar_comp(comp), axis=1)
        dados['treinamento'] = pd.merge(dados['treinamento'], dados['treinamento_pteu'],on='index_text', how='inner').drop(['grade', 'index_text', 'id'], axis=1)
        dados['validacao']['competencia'] = dados['validacao'].apply(selecionar_comp(comp), axis=1)
        dados['validacao'] = pd.merge(dados['validacao'], dados['validacao_pteu'],on='index_text', how='inner').drop(['grade', 'index_text', 'id'], axis=1)
        if tipo == 'nao_neurais_merge':
            treinamento = pd.concat([dados['treinamento'], dados['validacao']], axis=0)
        else:
            treinamento = dados['treinamento']
    else:
        treinamento = dados['treinamento']
    return treinamento

def get_modelo(regressor, params):
    if regressor == 'LR':
        model = LinearRegression()
    elif regressor == 'RF':
        model = RandomForestRegressor(**params)
    elif regressor == 'tree':
        model = tree.DecisionTreeRegressor(**params)
    elif regressor == 'boosting':
        model = GradientBoostingRegressor(**params)
    elif regressor == 'xgboosting':
        model = XGBRegressor(**params)
    else:
        model = SVR(kernel='poly', **params)
    return model

def get_regressor(comp, regressor, tipo, params={}, selecionar=[]):
    treinamento = arrumar_dados_treinamento(comp, tipo)
    model = get_modelo(regressor, params)
    y = treinamento['competencia']
    if selecionar == []:
        x = treinamento.drop("competencia", axis=1).values
    else:
        x = treinamento[selecionar].values
    model.fit(x,y)
    return model


def get_feature_selector(comp, regressor, tipo, params_regressor, params, selecionar=[]):
    seletor = params['seletor']
    treinamento = arrumar_dados_treinamento(comp, tipo)
    model1 = get_modelo(regressor, params_regressor)
    if seletor == "SFS":
        model = SequentialFeatureSelector(model1, n_features_to_select=params['n_features_to_select'], n_jobs=-1, direction=params['direction'])
    elif seletor == "kbest":
        model = SelectKBest(f_regression, k=params['n_features_to_select'])
    y = treinamento['competencia']
    novo_treinamento = treinamento.drop("competencia", axis=1)
    if selecionar == []:
        x = novo_treinamento.values
    else:
        novo_treinamento = novo_treinamento[selecionar]
        x = novo_treinamento[selecionar].values
    model.fit(x,y)
    selected_features = model.get_support()
    return list(novo_treinamento.columns[selected_features])



def get_novos_dados(comp, tipo):
    pairs = {'C1': 'formal_register', 'C2': 'thematic_coherence', 'C3': 'narrative_rhetorical_structure', 'C4': 'cohesion'}
    enem = f'C{comp}'
    nome_bert_classification = f"Bert-classification-{pairs[enem]}--Narrativo.csv"
    nome_bert_regression = f"Bert-regression-{pairs[enem]}--Narrativo.csv"
    test = pd.read_csv("dataset\\testcleanNarrativoFeatures.csv", encoding='utf-8')
    test_bert_classification = pd.read_csv(f"dataset\\test{nome_bert_classification}", encoding='utf-8')
    test_bert_regression = pd.read_csv(f"dataset\\test{nome_bert_regression}", encoding='utf-8')
    dados = {}
    dados['teste'] = test
    dados['teste']['competencia'] = dados['teste'].apply(selecionar_comp(comp), axis=1)
    if tipo == 'bert_classification':
        dados['teste'] = pd.merge(dados['teste'], test_bert_classification, on='index_text')
        dados['teste'] = dados['teste'].drop(['grade', 'index_text', 'confidence_0', 'confidence_1', 'confidence_2', 'confidence_3', 'confidence_4', 'confidence_5'], axis=1)
    elif tipo in ['neurais', 'neurais_merge']:
        dados = get_dados(comp)
        #to aqui
        colunas = dados['validacao'].columns
        dados['validacao']['competencia'] = dados['validacao'].apply(selecionar_comp(comp), axis=1)
        dados['validacao'] = pd.merge(dados['validacao'], dados['validacao_bert_regression'], on='index_text', how='inner')
        dados['validacao'] = pd.merge(dados['validacao'], dados['validacao_bert_classification'], on='index_text', how='inner')
        dados['validacao'] = dados['validacao'].drop(colunas, axis=1)
        #na de cima
        dados['teste']['competencia'] = dados['teste'].apply(selecionar_comp(comp), axis=1)
        dados['teste'] = pd.merge(dados['teste'], dados['teste_bert_regression'], on='index_text', how='inner')
        dados['teste'] = pd.merge(dados['teste'], dados['teste_bert_classification'], on='index_text', how='inner')
        dados['teste'] = dados['teste'].drop(colunas, axis=1)
    elif tipo in ['iR4S', 'iR4S_merge',  'iR4S-F', 'iR4S-F_merge']:
        dados = get_dados(comp)
        colunas = list(dados['teste'].columns)
        colunas.remove('index_text')
        colunas.remove('grade')
        colunas.remove('pronoun_ratio')
        colunas.remove('ttr')
        dados['teste']['competencia'] = dados['teste'].apply(selecionar_comp(comp), axis=1)
        dados['teste'] = pd.merge(dados['teste'], dados['teste_pteu'], on='index_text', how='inner').drop(['id', 'grade', 'index_text', 'ttr_x', 'pronoun_ratio_x']+colunas, axis=1)
        if tipo in ['iR4S-F_merge', 'iR4S-F']:
            filtrar = filtrar_colunas(dados['teste'].columns)
            dados['validacao'] = dados['teste'].drop(filtrar, axis=1)
            dados['teste'] = dados['teste'].drop(filtrar, axis=1)
    elif tipo in ['pteu_full', 'pteu_full_sem_merge']:
        dados = get_dados(comp)
        dados['teste']['competencia'] = dados['teste'].apply(selecionar_comp(comp), axis=1)
        dados['teste'] = pd.merge(dados['teste'], dados['teste_pteu'],on='index_text', how='inner')
        dados['teste'] = pd.merge(pd.merge(dados['teste'], dados['teste_bert_regression'], on='index_text', how='inner'), dados['teste_bert_classification'], on='index_text', how='inner').drop(['grade', 'index_text', 'id'], axis=1)
        dados['validacao']['competencia'] = dados['validacao'].apply(selecionar_comp(comp), axis=1)
        dados['validacao'] = pd.merge(dados['validacao'], dados['validacao_pteu'],on='index_text', how='inner')
        dados['validacao'] = pd.merge(pd.merge(dados['validacao'], dados['validacao_bert_regression'], on='index_text', how='inner'), dados['validacao_bert_classification'], on='index_text', how='inner').drop(['grade', 'index_text', 'id'], axis=1)
    elif tipo in ['nao_neurais', 'nao_neurais_merge']:
        dados = get_dados(comp)
        dados['teste']['competencia'] = dados['teste'].apply(selecionar_comp(comp), axis=1)
        dados['teste'] = pd.merge(dados['teste'], dados['teste_pteu'],on='index_text', how='inner').drop(['grade', 'index_text', 'id'], axis=1)
        dados['validacao']['competencia'] = dados['validacao'].apply(selecionar_comp(comp), axis=1)
        dados['validacao'] = pd.merge(dados['validacao'], dados['validacao_pteu'],on='index_text', how='inner').drop(['grade', 'index_text', 'id'], axis=1)
    else:
        dados['teste'] = dados['teste'].drop(['grade', 'index_text'], axis=1)
    return dados
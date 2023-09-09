import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pickle


# importa o modelo treinado
modelo_salvo = None
with open('modelo.pkl', 'rb') as arquivo:
    modelo_salvo = pickle.load(arquivo)


# função que formata os parâmetros de entrada para o modelo e retorna o valor da renda
def previsao_renda(dados: dict) -> float:
    sexo = 1. if dados['sexo'] == 'Feminino' else 0.
    posse_de_veiculo = 1. if dados['posse_de_veiculo'] == True else 0.
    posse_de_imovel = 1. if dados['posse_de_imovel'] == True else 0.
    
    faixa_tempo_emprego_6_11 = 0.
    faixa_tempo_emprego_12_17 = 0.
    faixa_tempo_emprego_18_23 = 0.
    faixa_tempo_emprego_24_29 = 0.
    faixa_tempo_emprego_30_35 = 0.
    faixa_tempo_emprego_36_41 = 0.
    faixa_tempo_emprego_42mais = 0.
    faixa_tempo_emprego_pensionista = 0.
    
    if dados['tipo_renda'] == 'Pensionista':
        faixa_tempo_emprego_pensionista = 1.
    elif (dados['tempo_emprego'] >= 6) & (dados['tempo_emprego'] <= 12):
        faixa_tempo_emprego_6_11 = 1.
    elif (dados['tempo_emprego'] >= 12) & (dados['tempo_emprego'] <= 17):
        faixa_tempo_emprego_12_17 = 1.
    elif (dados['tempo_emprego'] >= 18) & (dados['tempo_emprego'] <= 23):
        faixa_tempo_emprego_18_23 = 1.
    elif (dados['tempo_emprego'] >= 24) & (dados['tempo_emprego'] <= 29):
        faixa_tempo_emprego_24_29 = 1.
    elif (dados['tempo_emprego'] >= 30) & (dados['tempo_emprego'] <= 35):
        faixa_tempo_emprego_30_35 = 1.
    elif (dados['tempo_emprego'] >= 36) & (dados['tempo_emprego'] <= 41):
        faixa_tempo_emprego_36_41 = 1.
    elif (dados['tempo_emprego'] >= 42):
        faixa_tempo_emprego_42mais = 1.

    tipo_renda_empresario = 0.
    tipo_renda_pensionista = 0.
    tipo_renda_servidorpublico = 0.
    
    if dados['tipo_renda'] == 'Pensionista':
        tipo_renda_pensionista = 1.
    elif dados['tipo_renda'] == 'Empresário':
        tipo_renda_empresario = 1.
    elif dados['tipo_renda'] == 'Servidor público':
        tipo_renda_servidorpublico = 1.
        
    faixa_etaria_Idoso = 0.
    faixa_etaria_Jovem = 0.
    faixa_etaria_Pensionista = 0.
    
    if dados['tipo_renda'] == 'Pensionista':
        faixa_etaria_Pensionista = 1.
    elif dados['idade'] <= 29:
        faixa_etaria_Jovem = 1.
    elif dados['idade'] >= 60:
        faixa_etaria_Idoso = 1.
        
    if (dados['educacao'] == 'Pós graduação') | (dados['educacao'] == 'Superior completo'):
        curso_superior = 1.
    else:
        curso_superior = 0.
    
    params = [
        1.,
        sexo,
        faixa_tempo_emprego_12_17,
        faixa_tempo_emprego_18_23,
        faixa_tempo_emprego_24_29,
        faixa_tempo_emprego_30_35,
        faixa_tempo_emprego_36_41,
        faixa_tempo_emprego_42mais,
        faixa_tempo_emprego_6_11,
        faixa_tempo_emprego_pensionista,
        tipo_renda_empresario,
        tipo_renda_pensionista,
        tipo_renda_servidorpublico,
        faixa_etaria_Idoso,
        faixa_etaria_Jovem,
        faixa_etaria_Pensionista,
        curso_superior,
        posse_de_veiculo,
        posse_de_imovel
    ]
    
    valor = modelo_salvo.predict(params)[0]
    return np.exp(valor)


# faz a leitura da base de dados para recuperar as opções 
renda = pd.read_csv('renda.csv')


# recupera as opções para a variável educação
list_educacao = list(renda['educacao'].unique())
list_educacao.append("")


# recupera as opções para a variável tipo de renda
list_tipo_renda = list(renda['tipo_renda'].unique())
list_tipo_renda.append("")


# idade
idade_min = 18
idade_max = renda['idade'].max()


# Configurações da página
st.set_page_config(
    page_title="Previsão de renda", 
    layout="wide", 
    initial_sidebar_state="expanded"
)
st.markdown("<style>.css-1544g2n {margin-top: -75px;} .css-z5fcl4 {margin-top: -75px;}</style>", unsafe_allow_html=True)


# Adicionando o menu lateral
st.sidebar.title('Seleção de dados')
st.sidebar.divider()
var_sexo = st.sidebar.selectbox("Sexo:", ['', 'Feminino', 'Masculino'])
var_idade = st.sidebar.number_input("Idade:", min_value=idade_min, max_value=idade_max)
var_educacao = st.sidebar.selectbox("Educação:", sorted(list_educacao))
var_tipo_renda = st.sidebar.selectbox("Tipo de renda:", sorted(list_tipo_renda))
var_tempo_emprego = st.sidebar.number_input("Tempo no emprego:", min_value=0, max_value=99)
var_possui_veiculo = st.sidebar.checkbox('Possui Veículo')
var_possui_imovel = st.sidebar.checkbox('Possui Imóvel')


st.markdown(f"<h1 style='text-align: center;'>Previsão de renda</h1>", unsafe_allow_html=True)
st.markdown(f"<div style='text-align: center;'>Selecione os dados no menu ao lado para realizar a previsão da renda</div>", unsafe_allow_html=True)
st.divider()


# Variáveis selecionadas pelo usuário na tela
check_params = [
    len(str(var_sexo)) > 0,
    (var_idade > 0),
    len(str(var_educacao)) > 0,
    len(str(var_tipo_renda)) > 0,
    (var_tempo_emprego >= 0),
]


# verifica se o usuário selecionou todas as Variáveis
existe_dado_nao_informado = any(isinstance(elemento, bool) and not elemento for elemento in check_params)


valor_renda = 0


if not existe_dado_nao_informado:
    params = {
        'sexo': var_sexo,
        'posse_de_veiculo': var_possui_veiculo,
        'posse_de_imovel': var_possui_imovel,
        'educacao': var_educacao,
        'tipo_renda': var_tipo_renda,
        'idade': var_idade,
        'tempo_emprego': var_tempo_emprego,
    }
    valor_renda = previsao_renda(params)



st.markdown(f"<h1 style='text-align: center; color: #6884FF'>R$ {valor_renda:0.2f}</h1>".replace('.', ','), unsafe_allow_html=True)
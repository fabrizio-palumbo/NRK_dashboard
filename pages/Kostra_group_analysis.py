# Importing the required libraries

import pandas as pd
import streamlit as st
st.set_page_config(layout="wide")
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import os
from matplotlib import cm, colors
from scipy.stats import mannwhitneyu, wilcoxon
from scipy.stats import pearsonr,spearmanr



#---------------------------------------------------------------------------------------------------------
# Loading common database directory 
cwd = os.getcwd()
db_folder=cwd+"/database/"

## load functions 

import json as js

def calculate_pvalues(df,type):
        dfcols = pd.DataFrame(columns=df.columns)
        pvalues = dfcols.transpose().join(dfcols, how='outer')
        for r in df.columns:
            for c in df.columns:
                tmp = df[df[r].notnull() & df[c].notnull()]
                if type=="pearsonr":
                    pvalues[r][c] = round(pearsonr(tmp[r], tmp[c])[1], 4)
                else:
                    if type=="spearman":
                        pvalues[r][c] = round(spearmanr(tmp[r], tmp[c])[1], 4)
        return pvalues

def plot_correlation_matrix(df2,type,significance=0.1, annotated=False):
    fig=plt.figure(figsize=(8,6))
    p_values = calculate_pvalues(df2,type)                     # get p-Value
    mask_sig = np.invert((p_values<significance))    # mask - only get significant corr
    sns.heatmap(df2.corr(type),mask=mask_sig, cmap='RdBu_r',annot=annotated,vmin=-1, vmax=1)
    st.pyplot(fig)
    return
def loadMap(jsonMap):
    KommuneMap=js.load(jsonMap)
    KommuneCoordinates=pd.DataFrame(gp.GeoDataFrame.from_features(KommuneMap["features"]));
    KommuneCoordinates=KommuneCoordinates.drop_duplicates(subset='Kommunenummer', keep="first")
    Map=gp.GeoDataFrame(KommuneCoordinates);
    return Map
def orderData(reference, df):
    df.index = df.index.map(str)
    reference=pd.DataFrame(reference)
    R=reference.merge(df.reset_index(),'inner')
    return R
def plotMap(data, ref, axis, color_mapping, **kwargs):
    A=orderData(ref,data)
    gp.GeoDataFrame(A).plot(A.columns[-1], cmap=color_mapping,ax=axis, 
                            legend=False)
    plt.gca().axes.get_xaxis().set_visible(False);
    plt.yticks([])
    plt.xticks([])
    return
def quartile_dataset(df,n=4):
    print(df)
    L=pd.DataFrame()
    L["label_quartiles"]= pd.qcut(df,q =n, labels = False)
    # L.index = L.index.map(str)
    print(df)
    return L
def stat_test(df):
    for col in range(0, len(df.columns)-1):
        col1=df.columns[col]
        col2=df.columns[col+1]
        stat, p =mannwhitneyu(df[col1].dropna(),df[col2].dropna(),alternative= "less") 
        #ranksums
        # interpret
        alpha = 0.1
        if p > alpha:
            st.write(df.columns[col], "-", df.columns[col+1], 'Same distribution (fail to reject H0), p', p)
        else:
            st.write(df.columns[col], "-", df.columns[col+1],'Different distribution (reject H0), p', p)
    paired_data=df[["2019","2020","2021"]].dropna(axis=0, how="any")
    for col in range(0, len(paired_data.columns)-1):
        stat, p =wilcoxon(paired_data.iloc[:,col],paired_data.iloc[:,col+1],alternative="less")
        alpha = 0.1
        if p > alpha:
            st.write(paired_data.columns[col], "-", paired_data.columns[col+1], 'Same distribution (fail to reject H0), Paired test' , "p=", p)
        else:
            st.write(paired_data.columns[col], "-", paired_data.columns[col+1],'Different distribution (reject H0), Paired test ' , "p=", p)
    return

#-------------------------------from fabrizio 
#Loading the dataset
data_komune_code =st.session_state.kom_kode
list_variables=st.session_state.variables 
data_kostra =st.session_state.kostra
years_list=["2016","2017","2018","2019","2020","2021"]
def main():
    year_selected = st.selectbox('Please select the year of interest',options= years_list,index=len(years_list)-2)     
    dataset=pd.DataFrame()
    dataset.index.name="komnr"
    var_names=[]
    for var in list_variables.keys():
        var_names.append(var)
            # st.write(len(list_variables[var][year_selected].dropna()))
        try:
            to_append=list_variables[var][year_selected].dropna()#.rename(index={'301':'0301'},inplace=True)
            dataset[var]=to_append
        except Exception as error:
            st.write("variable ", var, "missing for year ", year_selected)    
            dataset[var]=np.nan
    dataset["kostragr"]=data_kostra.kostragr.astype(int)
    pairplot_container = st.container()
    col1corr, col2pair = st.columns([4,4])
    with pairplot_container:
        with col1corr:
            with st.form(key='kostra_corr'):
                labels = st.checkbox('Display values')
                q_val= st.selectbox(
                "select quantile of interest per Kostra group",
                [0,0.25,0.5,0.75],
                2)
                st.write("Correlation analysis of Quantiles of interest per Variable per Kostra Group")
                st.write("Kostra Group 16 removed because of lack of data (4 kommuner < 600 inhabitants) ")
                dataset_Kostra=dataset.query("kostragr !=16 ").groupby(by=['kostragr'], axis=0, level=None, as_index=True, sort=False,dropna=True).quantile(q_val)
                plot_correlation_matrix(dataset_Kostra,"spearman",annotated=labels)
                submit_button_kostra_pairplot = st.form_submit_button(label='Submit')
        with col2pair:     
            with st.form(key='kostra_pairplot'):
                options = st.multiselect(
                "select variable of interest to further visualize",
                list_variables.keys(),
                ["Med_ncr","Stillingsstørrelse"])
                agree = st.checkbox('Remove oslo')
                url = "https://www.ssb.no/en/klass/klassifikasjoner/112/koder"
                st.write("Info about Kostra grouping (%s)" % url)
                if agree:
                    fig_pairplot=sns.pairplot(dataset_Kostra[options].drop(13),kind="reg", plot_kws={'line_kws':{'color':'red'}})
                else:
                    fig_pairplot=sns.pairplot(dataset_Kostra[options],kind="reg", plot_kws={'line_kws':{'color':'red'}})
                #st.write(dataset_Kostra)
                submit_button_kostra_plot = st.form_submit_button(label='Submit')
            st.pyplot(fig_pairplot)
    
    with st.form(key='detailed_quartile_plot'):
                    var_to_explore= st.selectbox(
                    "select variable for kostra quartile selections",
                    var_names,
                    var_names.index("Med_ncr"))
                    st.write("Use this variable to select all the municipalities per kostragroup in the quartile selected above (correlation matrix). I.E. if Med_ncr you are selection the top 50% (if quantile =0.5) municipalities per medcical ncr.")
                    x_to_plot= st.selectbox(
                    "select variable x to plot",
                    var_names,
                    var_names.index("Stillingsstørrelse"))
                    y_to_plot= st.selectbox(
                    "select variable y to plot",
                    var_names,
                    var_names.index("Med_ncr"))
                    submit_button_detailed_plot = st.form_submit_button(label='Submit')
    data_kostra_raw_index=[]
    for ind in dataset.index:
        ref=dataset["kostragr"][ind]
        if ref !=16:
            ref_val=dataset_Kostra.loc[ref][var_to_explore].item()
            value_k=dataset[var_to_explore][ind]
            if(value_k>=ref_val):  
                data_kostra_raw_index.append(ind)
            data_kostra_raw=dataset.loc[data_kostra_raw_index]
    pairplot_container = st.container()
    col1pair, col2pair = st.columns([4,4])
    with pairplot_container:
        with col2pair:
            sc=plt.figure()
            #sc=sns.pairplot(data_kostra_raw[options],kind="reg", plot_kws={'line_kws':{'color':'red'}})
            # sc=sns.lmplot(data=data_kostra_raw,x= x_to_plot, y= y_to_plot,truncate=True,robust=True, order=1)#

            tmptest = data_kostra_raw[data_kostra_raw[x_to_plot].notnull() & data_kostra_raw[y_to_plot].notnull()]
            r, pvalue = spearmanr(tmptest[x_to_plot], tmptest[y_to_plot])
            sns.regplot(data=data_kostra_raw,x= x_to_plot, y= y_to_plot,truncate=True,robust=True, order=1,label=f'Spearman = {r:.2f}, pval= {pvalue:.2f}')#
            plt.legend()
            st.pyplot(sc)
            with col1pair:
                plot_correlation_matrix(data_kostra_raw,"spearman")

# calling main function
main()


    



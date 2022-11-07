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
import geo_plot as gp
from matplotlib import cm, colors

cwd = os.getcwd()
db_folder=cwd+"/database/"

import json as js
import geopandas as gp
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
def quartile_dataset(df):
    print(df)
    L=pd.DataFrame()
    L["label_quartiles"]= pd.qcut(df,q = 4, labels = False)
    # L.index = L.index.map(str)
    print(df)
    return L
#Loading the dataset
data_komune_code = pd.read_csv(db_folder+ 'Komune_Kode.csv',encoding='latin-1')
data_arsvekt_per_user = pd.read_csv(db_folder+ 'Årsvekt_per_user.csv',encoding='latin-1',index_col=0)
data_education = pd.read_csv(db_folder+ 'education_level.csv', encoding='latin-1', index_col='komnr')
data_ed_percentage = pd.read_csv(db_folder+ 'education_percentage.csv', encoding='latin-1', index_col='komnr')
data_educationH= pd.read_csv(db_folder+ 'education_High.csv', encoding='latin-1', index_col='komnr')
data_educationL = pd.read_csv(db_folder+ 'education_low.csv', encoding='latin-1', index_col='komnr')

data_earnering=pd.read_csv(db_folder+ 'earnering.csv',encoding='utf-8')
data_befolkning_69 = pd.read_csv(db_folder+ 'befolkning_69.csv',encoding='latin-1',index_col='komnr')
data_heltid = pd.read_csv(db_folder+ 'heltid.csv',encoding='latin-1',index_col=0)
#data_årsvekt = pd.read_csv(db_folder+ 'Årsvekt.csv',encoding='utf-8',index_col=0)
data_lonn = pd.read_csv(db_folder+ 'lonn.csv',encoding='latin-1',index_col=0)
# data_lonn.index = data_lonn.index.map(str)
data_plass_list = pd.read_csv(db_folder+ 'plass_list.csv',encoding='latin-1',index_col=0)
data_stilstor = pd.read_csv(db_folder+ 'stilstor.csv',encoding='latin-1',index_col=0)

data_timar_i_uke = pd.read_csv(db_folder+ 'timar_i_uka.csv',encoding='latin-1',index_col='komnr')
data_timar_i_uke.index = data_timar_i_uke.index.map(int)
data_timar_i_uke_67plus = pd.read_csv(db_folder+ 'timar_i_uka_67plus.csv',encoding='latin-1',index_col='komnr')
data_timar_i_uke_67plus.index = data_timar_i_uke_67plus.index.map(int)


data_users = pd.read_csv(db_folder+ 'users.csv',encoding='latin-1',index_col='komnr')
data_users_over_67 = pd.read_csv(db_folder+ 'users_over_67.csv',encoding='latin-1',index_col=0)
data_vakter = pd.read_csv(db_folder+ 'vakter.csv',encoding='latin-1',index_col='komnr')
data_kostra = pd.read_csv(db_folder+ 'kostra_group.csv',encoding='latin-1', index_col='komnr')
data_kpr = pd.read_csv(db_folder+ 'kpr.csv',encoding='utf-8')

# data_kostra.index = data_kostra.index.map(str)
data_all_ncr = pd.read_csv(db_folder+ 'all_ncr.csv',encoding='latin-1',index_col=0)
data_all_ncr=data_all_ncr.apply(pd.to_numeric, errors='coerce')
# data_all_ncr["komnr"]=data_all_ncr["komnr"].astype(str)
data_all_ncr=data_all_ncr.groupby(by=['komnr'], axis=0, level=None, as_index=True, sort=False,dropna=True).sum()

#data_all_ncr.set_index('komnr',drop=True, append=False, inplace=True, verify_integrity=True)

# data_med_ncr = pd.read_csv(db_folder+ 'med_ncr.csv',encoding='latin-1',index_col=0)
# data_med_ncr=data_med_ncr.apply(pd.to_numeric, errors='coerce')
# # data_med_ncr["komnr"]=data_med_ncr["komnr"].astype(str)
# data_med_ncr=data_med_ncr.groupby(by=['komnr'], axis=0, level=None, as_index=True, sort=False,dropna=True).sum()#.set_index('komnr',drop=True, append=False, inplace=True, verify_integrity=True)

jsonMap_of_norway=open(db_folder+ "kommuner2021.json");
norwayMap= loadMap(jsonMap_of_norway)
norwayMap= norwayMap.rename(columns={"Kommunenummer": "komnr"})
kommuneMap=(norwayMap["komnr"])

def cluster_corr_data(data, num_of_clusters,metod,name_variable, fig_size=(8,6), heatmap_plot=False):
    from scipy.stats import zscore
    data=data.dropna(axis=0,how="all").dropna(axis=1,how="all")#.replace('.',0).replace(np.nan, 0).astype(int)
    corrMatrix=data.T.corr()
    # st.write(data)
    cg = sns.clustermap(corrMatrix.replace(np.nan, 0),cmap ="YlGnBu",method=metod);
    cluster_link=cg.dendrogram_col.linkage
    from scipy.cluster import hierarchy
    clust_num=hierarchy.cut_tree(cluster_link, num_of_clusters)
    lut1 = dict(zip(np.unique(clust_num), sns.hls_palette(len(np.unique(clust_num)), l=0.5, s=0.8)))
    row_colors1 = pd.DataFrame(clust_num)[0].map(lut1)
    if(heatmap_plot):
        cg2=sns.clustermap(corrMatrix.replace(np.nan, 0).reset_index(drop=True),cmap ="coolwarm",method=metod,
        row_colors=row_colors1,yticklabels=False, xticklabels=False, robust=True,dendrogram_ratio=(0.15,0),figsize=fig_size)
        st.pyplot(cg2)
    label=[]
    fig, axs = plt.subplots(1, sharex=True,figsize=fig_size)
    index=[]
    traces_zscored=[]
    col_clust=pd.DataFrame(list(range(0,num_of_clusters)))[0].map(lut1)
    for n in range(0,num_of_clusters):
        kom_data=data.iloc[np.where(clust_num==n)[0]]
        index.append([kom_data.index])
        kom_zscore=kom_data.apply(zscore,axis=1)
        kom_mean= kom_zscore.mean(axis=0)
        label.append("cluster number: " + str(n))
        kom_mean.plot(color = col_clust[n], ax=axs)
        traces_zscored.append(kom_mean)
        axs.legend(label)
        t=axs.set_title("z-scored traces averaged")
        plt.setp(t, color='w')    
        traces_zscored.append(kom_mean)
    st.pyplot(fig)
    cl_n=clust_num[:,0]
    d_out = {name_variable : pd.Series(cl_n,
                       index =data.index)}
    list_index=pd.DataFrame(d_out)
    list_index.index.name="komnr"
    list_index.columns=["Cluster #"]
    return index,list_index,col_clust,traces_zscored,row_colors1
def plot_correlation_matrix(df2,type):
    fig=plt.figure(figsize=(8,6))
    sns.heatmap(df2.corr(type),cmap='RdBu_r',annot=False,vmin=-1, vmax=1)
    st.pyplot(fig)
    return
list_variables={ "Education Ratio":data_education,
"Education %":data_ed_percentage,
"Education High":data_educationH.divide(data_users),
"Education Low":data_educationL.divide(data_users),
"Stilstor":data_stilstor,
"Timar i uka":data_timar_i_uke,
"Timar i uka 67+":data_timar_i_uke_67plus,
"Åarsvekt per user":data_arsvekt_per_user,
"heltid":data_heltid,
"Vakter":data_vakter.divide(data_users),
"Lonn":data_lonn.divide(data_users).dropna(axis=1, how="all"),
"User over 67":data_users_over_67.divide(data_users),
"Plass avaiable": data_plass_list ,
"Users total":data_users,
"All ncr":data_all_ncr.divide(data_users),
"Med ncr":data_med_ncr.divide(data_users)}
years_list=["2020","2021","2019"]

#Line graph function based on komune code
def stat_test(df):
    from scipy.stats import mannwhitneyu, wilcoxon
    for col in range(0, len(df.columns)-1):
        stat, p =mannwhitneyu(df.iloc[:,col].dropna(),df.iloc[:,col+1].dropna())
        # interpret
        alpha = 0.05
        if p > alpha:
            st.write(df.columns[col], "-", df.columns[col+1], 'Same distribution (fail to reject H0)')
        else:
            st.write(df.columns[col], "-", df.columns[col+1],'Different distribution (reject H0)')
    paired_data=df[["2019","2020","2021"]].dropna(axis=0, how="any")
    for col in range(0, len(paired_data.columns)-1):
        stat, p =wilcoxon(paired_data.iloc[:,0],paired_data.iloc[:,1],zero_method="pratt",alternative="less")
        alpha = 0.05
        if p > alpha:
            st.write(paired_data.columns[col], "-", paired_data.columns[col+1], 'Same distribution (fail to reject H0), Paired test' , "p=", p)
        else:
            st.write(paired_data.columns[col], "-", paired_data.columns[col+1],'Different distribution (reject H0), Paired test ' , "p=", p)
    return

def main():
    ncr_visualization = st.checkbox('Visualize ncr data')
    if ncr_visualization:
        ncr_all_violin=plt.figure()
        data_all_ncr_norm=data_all_ncr.divide(data_users)
        data_med_ncr_norm=data_med_ncr.divide(data_users)
        df_long_all=pd.wide_to_long(data_all_ncr_norm.reset_index(), stubnames='', i="komnr", j='year').reset_index().dropna()
        df_long_all.columns=["komnr","year","Ncr_ratio"]
        df_long_all["type"]="Total"
        df_long_all["year"]=df_long_all["year"].astype(str)
        df_long_med=pd.wide_to_long(data_med_ncr_norm.reset_index(), stubnames='', i="komnr", j='year').reset_index().dropna()
        df_long_med.columns=["komnr","year","Ncr_ratio"]
        df_long_med["year"]=df_long_med["year"].astype(str)
        df_long_med["type"]="Medicine"
        merged_long_df=pd.concat([df_long_med,df_long_all], axis=0)
        merged_long_df=merged_long_df.query("Ncr_ratio.notna() and year in @years_list ", engine="python")
        #st.write(merged_long_df)
        sns.boxplot(data=merged_long_df,x="year",y="Ncr_ratio",hue="type")#,split=True,inner="quart", linewidth=1,cut=0  )
        #plt.ylim([0,1])
        title_container = st.container()
        col1, col2 = st.columns([4,2])
        with title_container:
            with col1:     
                st.pyplot(ncr_all_violin)
            with col2:
                st.write("NCR Medicine statistical test across years")
                stat_test(data_med_ncr_norm)
                st.write("NCR Total statistical test across years")
                stat_test(data_all_ncr_norm)
        
       

        
    agree = st.checkbox('Visualize Kpr data')
    if agree:
        data_kpr["aar"]=data_kpr["aar"].astype(str)
        year_selected = st.selectbox('Please select the year of interest',options= set(data_kpr["aar"]))
        
        type_selected = st.selectbox('Please select the type of interest',options= set(data_kpr["tjenestetype - tekst"]))  
        st.write(type_selected)
        options_kpr = st.multiselect(
        "select variable of interest to average as KPR score",
        set(data_kpr["funksjonstype - tekst"]),
        list(set(data_kpr["funksjonstype - tekst"]))[0])
  
        kpr=data_kpr.query(" `funksjonstype - tekst` in @options_kpr and aar==@year_selected and `tjenestetype - tekst`==@type_selected")
        kpr=kpr.groupby(by=['kommunenummer','aar'], axis=0, level=None, as_index=False, sort=False,dropna=True).mean()
        list_variables.update({"KPR": kpr.pivot(index="kommunenummer", columns="aar", values="mean_value")})
   
    Display_earnering = st.checkbox('Visualize Earnering data:')
    if Display_earnering :
        data_earnering["Tidsperiode"]=data_earnering["Tidsperiode"].astype(str)
        data_earnering["komnr"]=data_earnering["komnr"].astype(int)
        year_selected_earnering = st.selectbox('Please select the year of interest',options= set(data_earnering["Tidsperiode"]))
        #st.write(data_earnering.index.duplicated())
        earnering_variable = st.selectbox('Please select the variable of interest',options= set(data_earnering["Måltall"]))
        earnering=data_earnering.query("Måltall== @earnering_variable")
        # st.write(earnering["komnr"])
        earnering=earnering.pivot(index="komnr", columns="Tidsperiode", values="Verdi")
        
        list_variables.update({"Earnering": earnering})
        #list_variables.update({"Earnering": earnering2021})
    
    st.write("Correlation analysis")
    year_selected = st.selectbox('Please select the year of interest',options= years_list)     
    dataset=pd.DataFrame()
    dataset.index.name="komnr"
    for var in list_variables.keys():
        # st.write(len(list_variables[var][year_selected].dropna()))
        try:
            to_append=list_variables[var][year_selected].dropna()#.rename(index={'301':'0301'},inplace=True)
            dataset[var]=to_append
        except Exception as error:
            st.write("variable ", var, "missing for year ", year_selected)

    dataset["kostragr"]=data_kostra.kostragr.astype(int)
    dataset_corr=dataset.iloc[:,:-1]
    plot_correlation_matrix(dataset_corr,"spearman")
    st.write("Correlation analysis of mean per Kostra Group")
    dataset_Kostra=dataset.groupby(by=['kostragr'], axis=0, level=None, as_index=True, sort=False,dropna=True).mean()
    plot_correlation_matrix(dataset_Kostra,"spearman")
    st.write("select varable of interest to further visualize")
    options = st.multiselect(
    "select varable of interest to further visualize",
    list_variables.keys(),
    ["Med ncr","Åarsvekt per user"])
    agree = st.checkbox('Remove oslo')
    if agree:
        fig_pairplot=sns.pairplot(dataset_Kostra[options].drop(13))
    else:
        fig_pairplot=sns.pairplot(dataset_Kostra[options])
    st.pyplot(fig_pairplot)
    label_quartiles_ncr_med=quartile_dataset(dataset[dataset["Users total"]>49]["Med ncr"].dropna())    
    P_data=pd.concat([dataset[options],label_quartiles_ncr_med],axis=1)
    list_quartiles= st.multiselect(
    "select varable of interest to further visualize",
    [0,1,2,3],
    [0,3])
    P_data_extreme=P_data.query("label_quartiles in @list_quartiles")
    g=sns.pairplot(P_data_extreme,hue="label_quartiles",palette='tab10')
    st.pyplot(g)
    return


    
main()


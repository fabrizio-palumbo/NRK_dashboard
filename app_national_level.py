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
st.cache(ttl=24*3600)
cwd = os.getcwd()
db_folder=cwd+"/database/"

import json as js

def calculate_pvalues(df,type):
        dfcols = pd.DataFrame(columns=df.columns)
        pvalues = dfcols.transpose().join(dfcols, how='outer')
        for r in df.columns:
            for c in df.columns:
                tmp = df[df[r].notnull() & df[c].notnull()]
                if type=="pearson":
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
#Loading the dataset
data_komune_code = pd.read_csv(db_folder+ 'Komune_Kode.csv',encoding='latin-1')
data_arsvekt_per_user = pd.read_csv(db_folder+ 'Årsvekt_per_user.csv',encoding='latin-1',index_col=0)
data_education = pd.read_csv(db_folder+ 'education_level.csv', encoding='latin-1', index_col='komnr')
data_ed_percentage = pd.read_csv(db_folder+ 'education_percentage.csv', encoding='latin-1', index_col='komnr')
data_educationH= pd.read_csv(db_folder+ 'education_High.csv', encoding='latin-1', index_col='komnr')
data_educationL = pd.read_csv(db_folder+ 'education_low.csv', encoding='latin-1', index_col='komnr')
data_users_very_sick = pd.read_csv(db_folder+ 'users_very_sick.csv',encoding='utf-8',index_col='komnr')
#data_users_very_sick.index=data_users_very_sick.index.map(int)
data_earnering=pd.read_csv(db_folder+ 'earnering.csv',encoding='utf-8')
data_befolkning_69 = pd.read_csv(db_folder+ 'befolkning_69.csv',encoding='latin-1',index_col='komnr')
data_heltid = pd.read_csv(db_folder+ 'heltid.csv',encoding='latin-1',index_col=0)
data_årsvekt = pd.read_csv(db_folder+ 'årsvekt.csv',encoding='utf-8',index_col=0)
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
data_all_ncr=data_all_ncr.groupby(by=['komnr'], axis=0, level=None, as_index=True, sort=False,dropna=True).sum(min_count=1)

#data_all_ncr.set_index('komnr',drop=True, append=False, inplace=True, verify_integrity=True)

data_med_ncr = pd.read_csv(db_folder+ 'med_NCR.csv',encoding='latin-1',index_col=0)
data_med_ncr=data_med_ncr.apply(pd.to_numeric, errors='coerce')
# data_med_ncr["komnr"]=data_med_ncr["komnr"].astype(str)
data_med_ncr=data_med_ncr.groupby(by=['komnr'], axis=0, level=None, as_index=True, sort=False,dropna=True).sum(min_count=1)#.set_index('komnr',drop=True, append=False, inplace=True, verify_integrity=True)

list_variables={ "All_ncr":data_all_ncr.divide(data_users),
"Med_ncr":data_med_ncr.divide(data_users),
"Users_total":data_users,
"Education_Ratio_(H/L)":data_education,
"%_High_educated_nurses":data_ed_percentage,
"Education_High":data_educationH.divide(data_users),
"Education_Low":data_educationL.divide(data_users),
"Stillingsstørrelse":data_stilstor,
"Timar_i_uka":data_timar_i_uke,
"Timar_ i_uka_67+":data_timar_i_uke_67plus,
"Åarsvekt_per_user":data_arsvekt_per_user,
"heltid":data_heltid,
"Vakter":data_vakter.divide(data_users).dropna(axis=1, how="all"),
"Lonn":data_lonn.divide(data_users).dropna(axis=1, how="all"),
"User_over_67":data_users_over_67.divide(data_users),
"Plass_avaiable": data_plass_list ,
"Users_very_sick": data_users_very_sick,
}
if 'variables' not in st.session_state:
    st.session_state['variables'] = list_variables
if 'kom_kode' not in st.session_state:
    st.session_state['kom_kode'] = data_komune_code
if 'kostra' not in st.session_state:
    st.session_state['kostra'] = data_kostra
    

years_list=["2016","2017","2018","2019","2020","2021"]

def main():
    ncr_visualization = st.checkbox('Visualize ncr data')
    if ncr_visualization:
        ncr_all_boxplot=plt.figure()
        data_med_ncr_norm=list_variables["Med ncr"]
        data_all_ncr_norm=list_variables["All ncr"]
        df_long_all=pd.wide_to_long(data_all_ncr_norm.reset_index(), stubnames='', i="komnr", j='year').reset_index().dropna()
        df_long_all.columns=["komnr","year","Ncr_ratio"]
        df_long_all["type"]="Total"
        df_long_all["year"]=df_long_all["year"].astype(str)
        df_long_med=pd.wide_to_long(data_med_ncr_norm.reset_index(), stubnames='', i="komnr", j='year').reset_index().dropna()
        df_long_med.columns=["komnr","year","Ncr_ratio"]
        df_long_med["year"]=df_long_med["year"].astype(str)
        df_long_med["type"]="Medicine"
        merged_long_df=pd.concat([df_long_med,df_long_all], axis=0)
        merged_long_df=merged_long_df.query("Ncr_ratio.notna() and year in @years_list ", engine="python").reset_index()
        sns.boxplot(data=merged_long_df,x="year",y="Ncr_ratio",hue="type")#,split=True,inner="quart", linewidth=1,cut=0  )
        plt.ylim([0,2])
        ncr_all_ci=plt.figure()
        sns.lineplot( data=merged_long_df.query("type=='Total'"),x="year",y="Ncr_ratio",color="g")
        plt.title("Total number of NCR")
        ncr_med_ci=plt.figure()
        sns.lineplot( data=merged_long_df.query("type=='Medicine'"),x="year",y="Ncr_ratio",color="b")
        plt.title("Medicine related NCR")

        
        title_container = st.container()
        col1, col2 , col3= st.columns([2,2,2])
        with title_container:
            with col1:     
                #st.pyplot(ncr_all_boxplot)
                st.write(merged_long_df.drop(["komnr","index"],axis=1).groupby(["year","type"], level=None, as_index=True).agg(['mean', 'std','median']))
            with col2:
                st.pyplot(ncr_all_ci)
            with col3:
                st.pyplot(ncr_med_ci)
                # st.write("NCR Medicine statistical test across years")
                # stat_test(data_med_ncr_norm)
                # st.write("NCR Total statistical test across years")
                # stat_test(data_all_ncr_norm)
    agreeKPR = st.checkbox('Visualize Kpr data')
    if agreeKPR:
        with st.form(key='kpr'):
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
            st.session_state.variables = list_variables
            submit_button_kpr = st.form_submit_button(label='Submit')

    Display_earnering = st.checkbox('Visualize Earnering data:')
    if Display_earnering :
        with st.form(key='earnering'):
            data_earnering["Tidsperiode"]=data_earnering["Tidsperiode"].astype(str)
            data_earnering["komnr"]=data_earnering["komnr"].astype(int)
            #year_selected_earnering = st.selectbox('Please select the year of interest',options= set(data_earnering["Tidsperiode"]))
            #st.write(data_earnering.index.duplicated())
            earnering_variable = st.selectbox('Please select the variable of interest',options= set(data_earnering["Måltall"]))
            earnering=data_earnering.query("Måltall== @earnering_variable")
      
            earnering=earnering.pivot(index="komnr", columns="Tidsperiode", values="Verdi")
            list_variables.update({"Earnering": earnering})
            st.session_state.variables = list_variables
            #list_variables.update({"Earnering": earnering2021})
            submit_button_earnering = st.form_submit_button(label='Submit')
    with st.form(key='all_data_filtering'):
        st.write("Correlation analysis")
        min_users= st.select_slider(
        'Select minimum number of patients per kommune',
        options=list(range(0,100,10)))
        paramters_container = st.container()
        col1p, col2p = st.columns([1,4])
        with paramters_container:
            with col1p:     
                var_percentiles=st.multiselect(
                "select variable of interest to filter data",
                list_variables.keys(),
                ["Med_ncr"],max_selections=1)
            with col2p:
                restrict_range_values= st.slider(
                'Select the percentiles of interest',
                0, 100, (0, 100),step=1)
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
        dataset=dataset.query("Users_total >= @min_users")
        low_p,high_p=restrict_range_values
        low_val=dataset[var_percentiles].quantile(low_p/100).item()
        high_val=dataset[var_percentiles].quantile(high_p/100).item()
        dataset=dataset.query("{0} >= @low_val & {0}  <= @high_val  ".format(var_percentiles[0]))

        dataset["kostragr"]=data_kostra.kostragr.astype(int)
        dataset_corr=dataset.iloc[:,:-1]
        submit_button_filter_all_data = st.form_submit_button(label='Submit')
    # st.write(dataset)
    corr_container = st.container()
    col1corr, col2corr = st.columns([4,4])
    with corr_container:
        with col1corr:     
            st.write("Correlation analysis of all kommuner together")
            plot_correlation_matrix(dataset_corr,"pearson")
        with col2corr:
            with st.form(key='all_data_pairplot'):
                options = st.multiselect(
                "select variable of interest to further visualize (first x, then y)",
                list_variables.keys(),
                ["Stillingsstørrelse","Med_ncr"],max_selections=2)
                # fig_all_pairplot=sns.pairplot(dataset_corr[options],kind="reg", plot_kws={'line_kws':{'color':'red'}, 'robust':True})
                tmp = dataset_corr[options][dataset_corr[options[0]].notnull() & dataset_corr[options[1]].notnull()]
                r, pvalue = pearsonr(tmp [options[0]], tmp [options[1]])
                fig_all_regplot=plt.figure()
                sns.regplot(data=dataset_corr,x=options[0], y= options[1],order=1,truncate=True,robust=True,label=f'pearson corr= {r:.2f}, pval= {pvalue:.2f}')#
                plt.legend()
                submit_all_data_pairplot = st.form_submit_button(label='Submit')
                st.pyplot(fig_all_regplot)
       
            
             
    pairplot_container = st.container()
    col1pair, col2pair = st.columns([4,4])
    with pairplot_container:
        with col1pair:
            with st.form(key='quartiles_plot'):
                var_quartiles= st.selectbox(
                "select variable for quartile calculation",
                var_names,
                var_names.index("Med_ncr"))
                n_quartile= st.selectbox(
                "select number of quantile calculation",
                range(1,11),
                1)
                label_quartiles=quartile_dataset(dataset[var_quartiles].dropna(),n_quartile)    
                P_data=pd.concat([dataset[options],label_quartiles],axis=1)
                list_quartiles= st.multiselect(
                "select quartile of interest to visualize",
                range(0,n_quartile),[0,n_quartile-1]
                )
                P_data_extreme=P_data.query("label_quartiles in @list_quartiles")
                g=sns.pairplot(P_data_extreme,hue="label_quartiles",palette='tab10')
                submit_button_earnering = st.form_submit_button(label='Submit')
            st.pyplot(g)          
        with col2pair:     
            var_to_explore= st.selectbox(
            "select variable for quantile exploration",
            P_data.columns[:-1],
            len(P_data.columns[:-1])-1)
            line_plot=plt.figure()
            sns.lineplot( data=P_data,x="label_quartiles",y=var_to_explore,color="b")
            plt.title("quartiles calculated on "+var_quartiles)
            st.pyplot(line_plot)
            a=P_data.query("label_quartiles==0")[var_to_explore].dropna()
            b=P_data.query("label_quartiles==1")[var_to_explore].dropna()
            stat, p =mannwhitneyu(a,b,alternative= "greater") 
            st.write(p)


    return


    
main()


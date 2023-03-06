
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import os
from matplotlib import cm, colors
# from io import BytesIO
cwd = os.getcwd()
db_folder=cwd+"/database/"
#import dataset form main page

list_variables_original=st.session_state.variables 

def cluster_corr_data(data, num_of_clusters,metod,name_variable, fig_size=(8,6), heatmap_plot=False, metric="euclidean"):
    corr_container = st.container()
    col1corr, col2corr = st.columns([4,4])
    with corr_container:
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
            with col1corr:     
                st.pyplot(cg2)
        label=[]
        fig, axs = plt.subplots(1, sharex=True,figsize=fig_size)
        index=[]
        traces_zscored=[]
        col_clust=pd.DataFrame(list(range(0,num_of_clusters)))[0].map(lut1)
        for n in range(0,num_of_clusters):
            kom_data=data.iloc[np.where(clust_num==n)[0]]
            index.append([kom_data.index])
            kom_zscore=kom_data#.apply(zscore,axis=1)
            kom_mean= kom_zscore.mean(axis=0)
            label.append("cluster number: " + str(n))
            kom_mean.plot(color = col_clust[n], ax=axs)
            traces_zscored.append(kom_mean)
            axs.legend(label)
            t=axs.set_title("z-scored traces averaged")
            plt.setp(t, color='w')    
            traces_zscored.append(kom_mean)
        with col2corr:     
            st.pyplot(fig)
        cl_n=clust_num[:,0]
        d_out = {name_variable : pd.Series(cl_n,
                        index =data.index)}
        list_index=pd.DataFrame(d_out)
        list_index.index.name="komnr"
        list_index.columns=["Cluster #"]
    return index,list_index,col_clust,traces_zscored,row_colors1
import copy

def main():   
    plot_container = st.container()
    col1, col2 = st.columns([4,4])
    
    with plot_container:
        variable_selected = st.multiselect(
            "select variable to investigate",
            [k for k in list_variables_original.keys()],
            [k for k in list_variables_original.keys()])
        year_selected = st.selectbox('Select year of interest',options= [k for k in ["2017","2018","2019","2020"]])
        with col1:
            list_variables=[]#pd.DataFrame()
            #list_variables.index.name="komnr"
            for var in variable_selected :
                list_variables=dict((k, list_variables_original[k]) for k in variable_selected )
                #try:
                    #st.write("here 1")
                    #st.write(list_variables_original[var])
                #    to_append=list_variables_original[var]#.rename(index={'301':'0301'},inplace=True)
                #    list_variables.extend(to_append)
                #except Exception as error:
                #    st.write("variable ", var, "missing for year ", year_selected)
            df=copy.deepcopy(list_variables)
            #st.write("here 2")

            #st.write(list_variables)
            for var in list_variables.keys():
                for year in list_variables[var].columns:

                    #st.write( list_variables[var][year].quantile(0.9))
                    #st.write( list_variables[var][year].quantile(0.1))

                    df[var][year].loc[list_variables[var][year] >= list_variables[var][year].quantile(0.8)] = 1
                    df[var][year].loc[list_variables[var][year] <= list_variables[var][year].quantile(0.2)] = -1
                    df[var][year].loc[(list_variables[var][year] > list_variables[var][year].quantile(0.2)) & (list_variables[var][year] < list_variables[var][year].quantile(0.8))] = 0
            #year_selected = st.selectbox('Select year of interest',options= [k for k in ["2020"]])
            dataset=pd.DataFrame()
            dataset.index.name="komnr"
            for var in list_variables.keys():
                #if (var not in ["Users_total", "Lonn", "User_over_67",'Plass_avaiable', 'Users_very_sick', 'Users_medium_to_very_sick']):
                try:
                    to_append=df[var][year_selected]#.dropna()#.rename(index={'301':'0301'},inplace=True)
                    dataset[var]=to_append
                except Exception as error:
                    st.write("variable ", var, "missing for year ", year_selected)    
                    dataset[var]=np.nan
            #print(dataset.keys())
        #with col2: 
        #a,b,c,d,e=cluster_corr_data(dataset, 6,"ward","name_variable", fig_size=(8,6), heatmap_plot=True,metric="hamming")
            fig, axs = plt.subplots(1, sharex=True)
            axs=sns.heatmap(dataset.sort_values(by=variable_selected),cmap="bwr")
            st.pyplot(fig)
    return


    
main()


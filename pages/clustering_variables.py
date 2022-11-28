
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

list_variables=st.session_state.variables 



import json as js
import geopandas as gp
def loadMap(jsonMap):
    KommuneMap=js.load(jsonMap)
    KommuneCoordinates=pd.DataFrame(gp.GeoDataFrame.from_features(KommuneMap["features"]));
    KommuneCoordinates=KommuneCoordinates.drop_duplicates(subset='Kommunenummer', keep="first")
    Map=gp.GeoDataFrame(KommuneCoordinates);
    return Map
def orderData(reference, df):
    df.index = df.index.map(int)
    df.index = df.index.map(str)
    st.write(df)

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
    #print(df)
    return L
jsonMap_of_norway=open(db_folder+ "kommuner2021.json");
norwayMap= loadMap(jsonMap_of_norway)
norwayMap= norwayMap.rename(columns={"Kommunenummer": "komnr"})
kommuneMap=(norwayMap["komnr"])

def cluster_corr_data(data, num_of_clusters,metod,name_variable, fig_size=(8,6), heatmap_plot=False):
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
            kom_zscore=kom_data.apply(zscore,axis=1)
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


def main():   
    variable_name = st.selectbox('Select the variable of interest',options= [k for k in list_variables.keys()])     
    number_of_cluster = st.selectbox('Select how many cluster to detect',options= range(2,11))     
    a,b,c,d,e=cluster_corr_data(list_variables[variable_name], number_of_cluster,"ward","name_variable", fig_size=(8,6), heatmap_plot=True)
    for i,element in enumerate(a):
        st.write("cluster number :", i,"contains ", np.asarray(element).size, "kommuner")
    new_cmap = colors.LinearSegmentedColormap.from_list('new_cmap',c,number_of_cluster)
    fig, axis = plt.subplots()
    plotMap(b,norwayMap, axis=axis, color_mapping=new_cmap)
    
    # buf = BytesIO()
    # fig.savefig(buf, format="png")
    # st.image(buf,width=100, use_column_width=100)
    plot_container = st.container()
    col1, col2 = st.columns([4,4])
    with plot_container:
        with col1: 
                st.pyplot(fig)

    return


    
main()


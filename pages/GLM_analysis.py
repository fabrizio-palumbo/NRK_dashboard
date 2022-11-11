
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import os
from matplotlib import cm, colors
import statsmodels.api as sm
from scipy import stats

list_variables=st.session_state.variables 
years_list=["2020","2021","2019"]
def main():   
    year_selected = st.selectbox('Please select the year of interest',options= years_list)     
    options= [k for k in list_variables.keys()]
    variable_regressor = st.selectbox('Select 1 variable to regress',options=["All ncr","Med ncr"])     
    min_users= st.select_slider(
    'Select minimum number of patients per kommune',
    options=list(range(0,100,10)))
    variable_to_remove=["Med ncr","All ncr"]
    remove_var = st.checkbox('Remove_variable')
    if remove_var:
        variable_selected = st.multiselect(
        "select variable to remove",
        list_variables.keys(),
        list_variables.keys())
        #variable_to_remove= st.selectbox('Select variable to remove',options=list_variables.keys())     
        variable_to_remove.extend(variable_selected)
    dataset=pd.DataFrame()
    dataset.index.name="komnr"
    for var in list_variables.keys():
        if var not in variable_to_remove:
            try:
                to_append=list_variables[var][year_selected].dropna()#.rename(index={'301':'0301'},inplace=True)
                dataset[var]=to_append
            except Exception as error:
                st.write("variable ", var, "missing for year ", year_selected)
    dataset[variable_regressor]=list_variables[ variable_regressor][year_selected].replace(0, np.nan).dropna()
    dataset=dataset.dropna()
    dataset=dataset.query(" `Users total` > @min_users")
    Xx=dataset.iloc[:,:-1]
    Yy=dataset.iloc[:,-1]
    Yy=Yy[:,np.newaxis]
    glm_binom = sm.GLM(Yy,Xx,family=sm.families.Gamma(link=sm.families.links.log()))#family=sm.families.Poisson(link=sm.families.links.log())
    res = glm_binom.fit()#method="lbfgs"
    st.write(res.summary())
    nobs = res.nobs
    y =(Yy)
    yhat = res.mu
    fig, ax = plt.subplots()
    ax.scatter(yhat, y)
    line_fit = sm.OLS(y, sm.add_constant(yhat, prepend=True)).fit()
    sm.graphics.abline_plot(model_results=line_fit, ax=ax)
    #st.write(res.aic)
    ax.set_title('Model Fit Plot')
    ax.set_ylabel('Observed values')
    ax.set_xlabel('Fitted values')

    fig2, ax2 = plt.subplots()

    ax2.scatter(yhat, res.resid_pearson)
    ax2.hlines(0,0,max(yhat))
    #ax.set_xscale("log")
    ax2.set_title('Residual Dependence Plot')
    ax2.set_ylabel('Pearson Residuals')
    ax2.set_xlabel('Fitted values')
    
    from scipy import stats

    fig3, ax3 = plt.subplots()

    resid = res.resid_deviance.copy()
    resid_std = stats.zscore(resid)
    ax3.hist(resid_std, bins=25)
    ax3.set_title('Histogram of standardized deviance residuals');
        
    from statsmodels import graphics
    fig4=graphics.gofplots.qqplot(resid, line='r')
    
    glm_container = st.container()
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with glm_container:
        with col1:
            st.pyplot(fig)
        with col2:
            st.pyplot(fig2)
        with col3:
            st.pyplot(fig3)
        with col4:
            st.pyplot(fig4)
    st.write("Tsne  Test")
    tsne_visualization = st.checkbox('Visualize Tsne')
    if tsne_visualization:
        from sklearn.manifold import TSNE
        test_tsne=dataset
        tsne = TSNE(  n_components= 2, n_iter=25000, n_iter_without_progress=50000,  perplexity=10 )#
        tsne_results = tsne.fit_transform(test_tsne)
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=20, min_samples=2).fit(tsne_results)
        df_subset=pd.DataFrame()
        df_subset['tsne-2d-one'] = tsne_results[:,0]
        df_subset['tsne-2d-two'] = tsne_results[:,1]
        df_subset['label']=Yy
        fig=plt.figure(figsize=(16,10))
        #plt.hist(np.log10(Yy))
        plt.scatter(x= tsne_results[:,0],y= tsne_results[:,1], c=Yy,cmap="hot" )#,vmin=-1, vmax=1
        plt.colorbar()
        st.pyplot(fig)
    PCA_visualization = st.checkbox('Visualize PCA')
    if PCA_visualization:
        from sklearn.preprocessing import StandardScaler
        N_of_pc= st.select_slider(
        'Select minimum number of patients per kommune',
        options=list(range(3,len(dataset))))
        # define scaler
        scaler = StandardScaler()
        #create copy of DataFrame
        scaled_df=dataset.copy()
        #created scaled version of DataFrame
        scaled_df=pd.DataFrame(scaler.fit_transform(scaled_df), columns=scaled_df.columns)
        from sklearn.decomposition import PCA

        #define PCA model to use
        pca = PCA(n_components=N_of_pc)

        #fit PCA model to data
        pca_fit = pca.fit_transform(scaled_df)
        PC_values = np.arange(pca.n_components_) + 1
        fig3=plt.figure()
        plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained')
        st.pyplot(fig3)
        loadings = pca.components_
        num_pc = pca.n_features_
        pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
        loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
        loadings_df['variable'] = dataset.columns.values
        loadings_df = loadings_df.set_index('variable')
        
        # output

        # positive and negative values in component loadings reflects the positive and negative 
        # correlation of the variables with the PCs. Except A and B, all other variables have 
        # positive projection on first PC.

        # get correlation matrix plot for loadings
        
        fig=plt.figure()
        ax = sns.heatmap(loadings_df, annot=True, cmap='Spectral')
        st.pyplot(fig)

        #plt.scatter(pca_fit[:,1],pca_fit[:,2])

        def myplot(score,coeff,labels=None):
            fig2=plt.figure(figsize=(16,10))
            xs = score[:,0]
            ys = score[:,1]
            n = coeff.shape[0]
            scalex = 1.0/(xs.max() - xs.min())
            scaley = 1.0/(ys.max() - ys.min())
            plt.scatter(xs * scalex,ys * scaley,s=5)
            for i in range(n):
                plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
                if labels is None:
                    plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'green', ha = 'center', va = 'center')
                else:
                    plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
        
            plt.xlabel("PC{}".format(1))
            plt.ylabel("PC{}".format(2))
            plt.grid()
            plt.xticks([])
            plt.yticks([])
            #plt.gca().axes.get_yaxis().set_visible(False)
            #plt.gca().axes.get_xaxis().set_visible(False)

            st.pyplot(fig2)
        myplot(pca_fit[:,0:2],np.transpose(pca.components_[0:2, :]),list(dataset.columns))







    return


    
main()


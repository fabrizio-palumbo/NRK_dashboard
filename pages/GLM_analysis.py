
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



# Yy=Yy[:,np.newaxis]
# glm_binom = sm.GLM(Yy,Xx,family=sm.families.Poisson(link=sm.families.links.log()))#family=sm.families.Poisson(link=sm.families.links.log()))
# #family=sm.families.Gamma(link=sm.families.links.log()))
# # #family = Gamma(link="log")
# # #family=sm.families.Poisson(link=sm.families.links.log())family=sm.families.Binomial()
# res = glm_binom.fit(method="lbfgs")
# print(res.summary())
# nobs = res.nobs
# y =(Yy)
# yhat = res.mu
# fig, ax = plt.subplots()
# ax.scatter(yhat, y)
# line_fit = sm.OLS(y, sm.add_constant(yhat, prepend=True)).fit()
# sm.graphics.abline_plot(model_results=line_fit, ax=ax)
# print(res.aic)

# ax.set_title('Model Fit Plot')
# ax.set_ylabel('Observed values')
# ax.set_xlabel('Fitted values')

list_variables=st.session_state.variables 
years_list=["2020","2021","2019"]
def main():   
    year_selected = st.selectbox('Please select the year of interest',options= years_list)     
    options= [k for k in list_variables.keys()]
    variable_regressor = st.selectbox('Select 1 variable to regress',options=["All ncr","Med ncr"])     
    dataset=pd.DataFrame()
    dataset.index.name="komnr"
    for var in list_variables.keys():
        if var not in ["All ncr","Med ncr"]:
            try:
                to_append=list_variables[var][year_selected].dropna()#.rename(index={'301':'0301'},inplace=True)
                dataset[var]=to_append
            except Exception as error:
                st.write("variable ", var, "missing for year ", year_selected)
    dataset[variable_regressor]=list_variables[ variable_regressor][year_selected].replace(0, np.nan).dropna()
    dataset=dataset.dropna()
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
    return


    
main()


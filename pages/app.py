# Importing the required libraries

from distutils.log import error
from logging import exception
from statistics import mean
from token import N_TOKENS
# from turtle import width
from matplotlib.font_manager import font_scalings
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import os

st.set_page_config(
    page_title="OsloMet AI Journal",
    page_icon="",
    layout="wide",
    
)


#---------------------------------------------------------------------------------------------------------
# Loading common database directory 
cwd = os.getcwd()
db_folder=cwd+"/database/"


#-------------------------------from fabrizio 
#Loading the dataset
data_komune_code =st.session_state.kom_kode
list_variables=st.session_state.variables 
data_kostra =st.session_state.kostra

#---------------------------------------------------------------------------------------------------------------
# Function for creating line plot
def plot_graph_kommune(dataframe_kom,dataframe_mean_kostra,kom_name,year,y_label):
    df_plot = pd.DataFrame({kom_name:dataframe_kom,'kostra_mean':dataframe_mean_kostra.mean(axis=0)
    ,'Year':  list(year)
    })

    band_plot = dataframe_mean_kostra.melt( value_vars=year, var_name="Year", value_name=y_label, col_level=None, ignore_index=True)    
    
    df_plot_kom_meankostra = df_plot.melt('Year', var_name='name', value_name=y_label)
   
    line = alt.Chart(df_plot_kom_meankostra).mark_line().encode(
    alt.X('Year',scale=alt.Scale(zero=False)),
    alt.Y(y_label,scale=alt.Scale(zero=False))
    ,color=alt.Color("name:N")
    )
        
    band = alt.Chart(band_plot).mark_errorband(extent='ci', color='orange'
    ).encode(
    x='Year',
    y= y_label,
    #color= "steelblue"
    )
    
    chart=alt.layer(band ,line).properties(
        height=250, width= 310
        ,autosize = 'pad'
      #   title=stock_title
    ).configure_title(
        fontSize=16
    ).configure_axis(
        titleFontSize=14,
        labelFontSize=12
    )
    return chart#line, band 


# -------------------------------------------------------------------------------------------------------  
# Main function 
def main():   
     
    # --------------------------------------------------------------------
    # Side bar 
    with st.sidebar:   

    # ------------------------------------------------------------------------
    # Koumne dropdown list 
        komune_name = st.selectbox('Select the komune name',options= [gruppetekst for gruppekode,gruppetekst in zip(data_komune_code['GRUPPEKODE'].unique(),data_komune_code['GRUPPETEKST'].unique())])     
        query_komune_name = data_komune_code.query("GRUPPETEKST == @komune_name")  
        komune_code = query_komune_name['GRUPPEKODE'].iloc[0]     
        kom_gruppe = data_kostra.loc[int(komune_code)]['kostragr']
        list_kom_kostra = list(data_kostra.query('kostragr == @kom_gruppe').index)
        list_komune_kostra = [str(w) for w in list_kom_kostra]  
        
        options = st.multiselect("Please select data",options = list_variables )
        
        if not options:
            options = list_variables  
        
  
    st.header("AI Investigative Journal Work") 
    # Row 1
    cols=[]
    cols.extend(st.columns(2))
    cols.extend(st.columns(3))
    cols.extend(st.columns(3))
    cols.extend(st.columns(3))
    cols.extend(st.columns(3))
    cols.extend(st.columns(3))

#     # col1,col2 = st.columns(2)
#     # Row 2
#     col3,col4,col5,col6 = st.columns(4)
#     # Row 3
#     col7,col8,col9,col10 = st.columns(4)
#     # Row 4
#     col11,col12,col13,col14 = st.columns(4)
#    # Row 5
#     col15,col16,col17,col18 = st.columns(4)
    
    for i,values in  enumerate(options):        
        
        data = list_variables[values] 
        # st.write(data)
        if values == 'Med ncr':
            with cols[0]:
                st.subheader(values)     
                try:
                    med_ncr = data.loc[int(komune_code)]                                
                    med_ncr_default_year = min(med_ncr[med_ncr.notnull()].index)                   
                    
                    count_med_index = 0 
                    for year in med_ncr.index:
                        if year >= med_ncr_default_year:
                            break             
                        else:
                            count_med_index += 1                                
                    
                    med_ncr_choice_year = st.selectbox("Select year", options = data.columns, key=3, index=count_med_index)
                        
                    count_med_year = 0 
                    for year in med_ncr.index:
                        if year == med_ncr_choice_year:
                            break
                        count_med_year += 1   
                    from_to_year = med_ncr.index[count_med_year:]         
                    med_ncr_from_to_year = med_ncr[from_to_year]
                    mean_med_ncr = data.loc[[int(komune_index) for komune_index in list_komune_kostra if int(komune_index) in data.index]]          
                    mean_from_to_year  = mean_med_ncr[from_to_year]
                    
                    chart_med_ncr = plot_graph_kommune(med_ncr_from_to_year,mean_from_to_year,komune_name,from_to_year, values)                        
                
                    chart_med_ncr     

                except Exception as error:
                    st.write("We miss some index value for this kom", komune_code, komune_name)
                    st.write(error.args)

        elif values == 'All ncr':
            with cols[1]:
                st.subheader(values)      
                try:
                    all_ncr = data.loc[int(komune_code)]                
                    all_ncr_default_year = min(all_ncr[all_ncr.notnull()].index)                  
                    count_all_ncr_index = 0 
                    for year in all_ncr.index:
                        if year >= all_ncr_default_year:
                            break
                        else:
                            count_all_ncr_index += 1  

                    all_ncr_choice_year = st.selectbox("Select year", options = data.columns, key=4,index=count_all_ncr_index)

                    count_ncr_year = 0 
                    for year in all_ncr.index:
                        if year == all_ncr_choice_year:
                            break
                        count_ncr_year += 1   

                    from_to_year_ncr = all_ncr.index[count_ncr_year:]  
                    all_ncr_from_to_year = all_ncr[from_to_year_ncr]          
                    mean_all_ncr = data.loc[[int(komune_index) for komune_index in list_komune_kostra if int(komune_index) in data.index]]
                    mean_all_ncr_from_to_year = mean_all_ncr[from_to_year_ncr]
                    chart_all_ncr = plot_graph_kommune(all_ncr_from_to_year,mean_all_ncr_from_to_year,komune_name,from_to_year_ncr,values)
                                       
                    chart_all_ncr 
                        
                except Exception as error: 
                    st.write("We miss some index value for this kom", komune_code, "Place Name :"+ komune_name,"->" + values)
                    st.write(error.args)

        else:            
                    
            try:
                dataset = data.loc[int(komune_code)]
                #mean_education=data_education.loc[list_komune_kostra].median(axis = 0)
                kostra_mean=data.loc[[ int(komune_index) for komune_index in list_komune_kostra if int(komune_index) in data.index]]#.median(axis = 0)
                
                line_plot=plot_graph_kommune(dataset,kostra_mean,komune_name,dataset.index, values)  
                with cols[i]:            
                    st.subheader(values)
                    line_plot
                # if values == 'Education Ratio':
                #     with col3:  
                #         st.subheader(values) 
                #         line_plot 

                # elif values == 'Stilstor':
                #     with col4:
                #         st.subheader(values)                   
                #         line_plot
                # elif values == 'Timar i uka':
                #     with col5:
                #         st.subheader(values)                   
                #         line_plot  
                # elif values == 'Timar i uka 67+':
                #     with col6:
                #         st.subheader(values)                   
                #         line_plot  
                # elif values == 'Åarsvekt per user':
                #     with col7:
                #         st.subheader(values)                   
                #         line_plot  
                # elif values == 'heltid':
                #     with col8:
                #         st.subheader(values)                   
                #         line_plot  
                # elif values == 'Vakter':
                #     with col9:
                #         st.subheader(values)                   
                #         line_plot  
                # elif values == 'Lonn':
                #     with col10:
                #             st.subheader(values)                   
                #             line_plot   
                # elif values == 'User over 67':
                #     with col11:
                #         st.subheader(values)                   
                #         line_plot  
                # elif values == 'Plass avaiable':
                #     with col12:
                #         st.subheader(values)                   
                #         line_plot  
                # elif values == 'Users total':
                #     with col13:
                #         st.subheader(values)                   
                #         line_plot                       
                # elif values == 'Education %':
                #     with col14:
                #         st.subheader(values)                   
                #         line_plot
                # elif values == 'Education High':
                #     with col15:
                #         st.subheader(values)                   
                #         line_plot    
                # elif values == 'Education Low':
                #     with col16:
                #         st.subheader(values)                   
                #         line_plot 
                # elif values == 'Users by sickness':
                #     with col17:
                #         st.subheader(values)                   
                #         line_plot    
                # elif values == 'Users very sick':
                #     with col18:
                #         st.subheader(values)                   
                #         line_plot          
            except Exception as error:
                st.write("We miss some index value for this kom", komune_code, "Place Name :"+ komune_name,"->" + values)
                st.write(error.args)  

# calling main function
main()


    



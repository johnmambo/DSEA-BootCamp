 # RFM Analysis
 RFM stands for Recency, Frequency and Monetary value.
 Is a technique used to quantitatively rand and group customers based on the recency, frequency and monetary total of their recent transactions to identify
 their buying behavours and perform targeted campaigns especially in the marketing domain.

 *Using RFM Analysis, a business can assess customers’:*
recency (How recent was the customer's last purchase?)
frequency (how often they make purchase in a given period?)
and monetary value (the amount spent on purchases)
Recency, Frequency, and Monetary value of a customer are three key metrics that provide information about customer engagement, loyalty, and value to a business and also help merketers make best of their advertising budget.

## RFM Analysis Using Python
### Key fields in calculating RFM
Customer IDs
Purchase Dates
Transaction Amounts

To perform RFM Analysis in Python, we need our data to have the above fields.

*Step 1 Imports the required Python Libraries*


'''
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.templates.default = "plotly_white"

'''

*Step 2 Import and Read the desired Dataset.*

'''
data = pd.read_csv("rfm_data.csv")
print(data.head())
'''

*Step 3 Calculate the RFM values*

'''
import datetime from datetime
'''
Calculate Recency by first subtracting the date from the current date and extract the number of days using the date.now().date() function.
It gives us the number of days since the customer's last purchase, representing their recency value

'''
data['PurchaseDate'] = pd.to_datetime(data['PurchaseDate'])
data['Recency'] = (datetime.now().date() - data['PurchaseDate'].dt.date).dt.days
'''
To Calculate Frequency, we group the data by 'customerID' and count the number of unique 'OrderID' values to determine the number of times a customer made purchases.

'''
frequency_data = data.groupby('CustomerID')['OrderID'].count().reset_index()
frequency_data.rename(columns={'OrderID': 'Frequency'}, inplace=True)
data = data.merge(frequency_data, on='CustomerID', how='left')
'''
To calculate the monetary value for each customer, we group the data by 'CustomerID' and summed the 'TransactionalAmount' values to calculate the total amount
spent by each customer. It gives us the monetary value, representing the total monetary contribution of each customer.

*Step 4 Calculate RFM Scores* 
Define the scoring creteria for each RFM value:

'''
recency_scores = [5, 4, 3, 2, 1]   *Higher score for lower recency (more recent)*
frequency_scores = [1, 2, 3, 4, 5]  *Higher score for higher frequency*
monetary_scores = [1, 2, 3, 4, 5]   *Higher score for higher monetary value*

'''
Calculate RFM scores
To calculate RFM scores, we used the pd.cut() function to divide recency, frequency, and monetary values into bins. 
We define 5 bins for each value and assign the corresponding scores to each bin.

'''
data['RecencyScore'] = pd.cut(data['Recency'], bins=5, labels=recency_scores)
data['FrequencyScore'] = pd.cut(data['Frequency'], bins=5, labels=frequency_scores)
data['MonetaryScore'] = pd.cut(data['MonetaryValue'], bins=5, labels=monetary_scores)
'''
*Step 5 RFM Value Segmentation*
To calculate the RFM score, we add the scores obtained for recency, frequency and monetary value. For example, if a customer has a recency score of 3, a frequency score of 4, and a monetary score of 5, their RFM score will be 12.
'''
data['RFM_Score'] = data['RecencyScore'] + data['FrequencyScore'] + data['MonetaryScore']
'''

Create RFM segments based on the RFM score
After calculating the RFM scores, we create RFM segments based on the scores. We divided RFM scores into three segments, namely “Low-Value”, “Mid-Value”, and “High-Value”. Segmentation is done using the pd.qcut() function, which evenly distributes scores between segments.
'''
segment_labels = ['Low-Value', 'Mid-Value', 'High-Value']
data['Value Segment'] = pd.qcut(data['RFM_Score'], q=3, labels=segment_labels)
'''

confirm the resulting Data if you have no errors
'''
print(data.head())
'''
Now let’s have a look at the segment distribution:

## RFM Segment Distribution
segment_counts = data['Value Segment'].value_counts().reset_index()
segment_counts.columns = ['Value Segment', 'Count']

pastel_colors = px.colors.qualitative.Pastel

### Create the bar chart
fig_segment_dist = px.bar(segment_counts, x='Value Segment', y='Count', 
                          color='Value Segment', color_discrete_sequence=pastel_colors,
                          title='RFM Value Segment Distribution')

*Update the layout*
fig_segment_dist.update_layout(xaxis_title='RFM Value Segment',
                              yaxis_title='Count',
                              showlegend=False)

*Show the figure*
fig_segment_dist.show()

The above segments that we calculated are RFM value segments. Now we’ll calculate RFM customer segments. The RFM value segment represents the categorization of customers based on their RFM scores into groups such as “low value”, “medium value”, and “high value”. These segments are determined by dividing RFM scores into distinct ranges or groups, allowing for a more granular analysis of overall customer RFM characteristics. The RFM value segment helps us understand the relative value of customers in terms of recency, frequency, and monetary aspects.

Now let’s create and analyze RFM Customer Segments that are broader classifications based on the RFM scores. These segments, such as “Champions”, “Potential Loyalists”, and “Can’t Lose” provide a more strategic perspective on customer behaviour and characteristics in terms of recency, frequency, and monetary aspects. Here’s how to create the RFM customer segments:

*Step 1 Create a new column for RFM Customer Segments*
'''
data['RFM Customer Segments'] = ''
'''

*step 2 Assign RFM segments based on the RFM score*
'''
data.loc[data['RFM_Score'] >= 9, 'RFM Customer Segments'] = 'Champions'
data.loc[(data['RFM_Score'] >= 6) & (data['RFM_Score'] < 9), 'RFM Customer Segments'] = 'Potential Loyalists'
data.loc[(data['RFM_Score'] >= 5) & (data['RFM_Score'] < 6), 'RFM Customer Segments'] = 'At Risk Customers'
data.loc[(data['RFM_Score'] >= 4) & (data['RFM_Score'] < 5), 'RFM Customer Segments'] = "Can't Lose"
data.loc[(data['RFM_Score'] >= 3) & (data['RFM_Score'] < 4), 'RFM Customer Segments'] = "Lost"
'''
print thr updated data with RFM Segments
'''
print(data[['CustomerID', 'RFM Customer Segments']])
'''

## RFM Analysis
Now let’s analyze the distribution of customers across different RFM customer segments within each value segment:

segment_product_counts = data.groupby(['Value Segment', 'RFM Customer Segments']).size().reset_index(name='Count')

segment_product_counts = segment_product_counts.sort_values('Count', ascending=False)

fig_treemap_segment_product = px.treemap(segment_product_counts, 
                                         path=['Value Segment', 'RFM Customer Segments'], 
                                         values='Count',
                                         color='Value Segment', color_discrete_sequence=px.colors.qualitative.Pastel,
                                         title='RFM Customer Segments by Value')
fig_treemap_segment_product.show()






*Now let’s analyze the distribution of RFM values within the Champions segment:*

Filter the data to include only the customers in the Champions segment
'''
champions_segment = data[data['RFM Customer Segments'] == 'Champions']

fig = go.Figure()
fig.add_trace(go.Box(y=champions_segment['RecencyScore'], name='Recency'))
fig.add_trace(go.Box(y=champions_segment['FrequencyScore'], name='Frequency'))
fig.add_trace(go.Box(y=champions_segment['MonetaryScore'], name='Monetary'))

fig.update_layout(title='Distribution of RFM Values within Champions Segment',
                  yaxis_title='RFM Value',
                  showlegend=True)

fig.show()
'''

Now let’s analyze the correlation of the recency, frequency, and monetary scores within the champions segment:
'''
correlation_matrix = champions_segment[['RecencyScore', 'FrequencyScore', 'MonetaryScore']].corr()
'''

*Visualize the correlation matrix using a heatmap*
'''
fig_heatmap = go.Figure(data=go.Heatmap(
                   z=correlation_matrix.values,
                   x=correlation_matrix.columns,
                   y=correlation_matrix.columns,
                   colorscale='RdBu',
                   colorbar=dict(title='Correlation')))

fig_heatmap.update_layout(title='Correlation Matrix of RFM Values within Champions Segment')

fig_heatmap.show()
'''

Now let’s have a look at the number of customers in all the segments:
'''
import plotly.colors

pastel_colors = plotly.colors.qualitative.Pastel

segment_counts = data['RFM Customer Segments'].value_counts()
'''
*
Create a bar chart to compare segment counts*
'''
fig = go.Figure(data=[go.Bar(x=segment_counts.index, y=segment_counts.values,
                            marker=dict(color=pastel_colors))])
'''

*Set the color of the Champions segment as a different color*
'''
champions_color = 'rgb(158, 202, 225)'
fig.update_traces(marker_color=[champions_color if segment == 'Champions' else pastel_colors[i]
                                for i, segment in enumerate(segment_counts.index)],
                  marker_line_color='rgb(8, 48, 107)',
                  marker_line_width=1.5, opacity=0.6)
'''

*Update the layout*
'''
fig.update_layout(title='Comparison of RFM Segments',
                  xaxis_title='RFM Segments',
                  yaxis_title='Number of Customers',
                  showlegend=False)

fig.show()
'''

### Now let’s have a look at the recency, frequency, and monetary scores of all the segments:

*Calculate the average Recency, Frequency, and Monetary scores for each segment*
'''
segment_scores = data.groupby('RFM Customer Segments')['RecencyScore', 'FrequencyScore', 'MonetaryScore'].mean().reset_index()
'''

*Create a grouped bar chart to compare segment scores*
'''
fig = go.Figure()
'''

 *Add bars for Recency score*
 '''
fig.add_trace(go.Bar(
    x=segment_scores['RFM Customer Segments'],
    y=segment_scores['RecencyScore'],
    name='Recency Score',
    marker_color='rgb(158,202,225)'
))
'''

 *Add bars for Frequency score*
 '''
fig.add_trace(go.Bar(
    x=segment_scores['RFM Customer Segments'],
    y=segment_scores['FrequencyScore'],
    name='Frequency Score',
    marker_color='rgb(94,158,217)'
))
'''

*Add bars for Monetary score*
'''
fig.add_trace(go.Bar(
    x=segment_scores['RFM Customer Segments'],
    y=segment_scores['MonetaryScore'],
    name='Monetary Score',
    marker_color='rgb(32,102,148)'
))
'''

 *Update the layout*
 '''
fig.update_layout(
    title='Comparison of RFM Segments based on Recency, Frequency, and Monetary Scores',
    xaxis_title='RFM Segments',
    yaxis_title='Score',
    barmode='group',
    showlegend=True
)

fig.show()
'''

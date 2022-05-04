#!/usr/bin/env python
# coding: utf-8

# # <u> AIML Module Project Submission  </u>

# In[1]:


# Importing all necessary libraries for statitical calculation and plotting
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import ttest_ind  
from statsmodels.stats.proportion import proportions_ztest  
import statsmodels.api         as     sm
from   statsmodels.formula.api import ols


# ## PART ONE : QUESTION BASED [Total Score : 15]

# ### Question 1: Please refer the table below to answer below questions:
# 
# |Planned to purchase Product A|Actually placed and order for Product A - Yes|Actually placed and order for Product A - No|Total|
# |------|------|------|------|
# |Yes  |400  |100  | 500 |
# |No  |200  |1300  | 1500 |
# |Total  |600  |1400  | 2000 |

# 1. Refer to the above table and find the joint probability of the people who planned to purchase and actually
# placed an order.

# In[2]:


# P_PYes : Probability of the people who planned to purchase the product A
# P_AYes : Probability of the people who actually purchased the product A

# P_A : Intersection of the probabilities P_PYes and P_AYes
# T_A : Total observations for product A

P_A = 400
T_A  = 2000
Joint_A = round(P_A / T_A,4) 

print("="*120)
print('Answer 1: Probability that people who planned to purchase , actually placed an order %1.3f' % Joint_A)
print("="*120)


# 2. Refer to the above table and find the joint probability of the people who planned to purchase and actually
# placed an order, given that people planned to purchase.

# In[3]:


# P1 = P(Planned to Purchase and Actually purchased)
P1   = 400 / 2000
# P2 = P(Planned to Purchase)
P2   = 500 / 2000
#P3  = P(Actually purchased | Planned to Purchase) = P(Planned to Purchase and Actually purchased) / P(Planned to Purchase)
P3   = P1 / P2

print("="*120)
print('Answer 2: P(Actually purchased | Planned to Purchase)  is %1.3f' % P3)    
print("="*120)


# ### Question 2: An electrical manufacturing company conducts quality checks at specified periods on the products it manufactures. Historically, the failure rate for the manufactured item is 5%. Suppose a random sample of 10 manufactured items is selected. 
# Answer the following questions.

# #### Notes:
# Here 
# * n = 10 ; number of random sample 
# * p = 0.05 ; probability of a defective product
#      
# We shall use binomial distribution as it is discrete random variable problem statement

# $P(X = x\mid n,p)$ = $\frac{n!}{x!(n - x)!}p^x (1 - p)^{n-x} $
# * where P(X = x) is the probability of getting x successes in n trials
# and $p$ is the probability of an event of interest

# A. Probability that none of the items are defective?

# In[4]:


p = 0.05
n = 10
x = 0
_10_C_0 = 1
P_0 = _10_C_0 *((p)**x) * ((1-p)**(n-x))

print("="*120)
print('Answer A: Probability that none of the items are defective %1.3f' % P_0)  
print("="*120)


# B. Probability that exactly one of the items is defective?

# In[5]:


p = 0.05
n = 10
_10_C_1 = 10
x = 1

P_1 = _10_C_1 * ((p)**x) * ((1-p)**(n-x))

print("="*120)
print('Answer B: Probability that exactly one of the items is defective %1.3f' % P_1) 
print("="*120)


# C. Probability that two or fewer of the items are defective?

# In[6]:


_10_C_0 = 1
x_0 = 0
_10_C_1 = 10
x_1 = 1
_10_C_2 = 45
x_2 = 2

P_LE2 = (_10_C_0 *((p)**x_0) * ((1-p)**(n-x_0))) + (_10_C_1 *((p)**x_1) * ((1-p)**(n-x_1))) + (_10_C_2 *((p)**x_2) * ((1-p)**(n-x_2)))

print("="*120)
print('Answer C: Probability that two or fewer of the items are defective %1.3f' % P_LE2) 
print("="*120)


# D. Probability that three or more of the items are defective ?

# In[7]:


# Optimizing the method of finding solution by using binomial distribution

p   =  0.05
n   =  10
k   =  np.arange(0,11)

binomial = stats.binom.pmf(k,n,p)
print(binomial)


# In[8]:


print("="*120)
print(' Answer A. Probability that none of the items are defective %1.4f' %binomial[0])
print("="*120)


# In[9]:


print("="*120)
print('Answer B. Probability that exactly one of the items is defective %1.4f' %binomial[1])
print("="*120)


# In[10]:


cumbinomial = stats.binom.cdf(k,n,p)
print(cumbinomial)


# In[11]:


print("="*120)
print('Answer C. Probability that two or fewer of the items are defective %1.4f' %cumbinomial[2])
print("="*120)


# In[12]:


P_3 = 1 - cumbinomial[2]

print("="*120)
print('Answer D. Probability that three or more of the items are defective %1.4f' %P_3)
print("="*120)


# ### Question 3: A car salesman sells on an average 3 cars per week.

# #### Notes : 
# By definition, the Poisson distribution is a discrete probability distribution for the counts of events that occur randomly in a given interval of time or space; which fits for this problem statement.
# 
# P(X = x) = $\frac{e^\lambda \lambda^x}{x!} $
# where 
# * P(x)              = Probability of x successes given an idea of  $\lambda$
# * $\lambda$ = Average number of successes
# * e                   = 2.71828 (based on natural logarithm)
# * x                    = successes per unit which can take values 0,1,2,3,... $\infty$

# In[13]:


rate =  3  # which is the average sale
n    =  np.arange(0,15)
poisson = stats.poisson.pmf(n,rate)
poisson


# A. Probability that in a given week he will sell some cars.

# In[14]:


# Probability of selling some cars is 1 - P(selling zero cars)
P_some = 1 - poisson[0]

print("="*120) 
print('Answer A. Probability that in a given week he will sell some cars %1.4f' %P_some)
print("="*120)


# B. Probability that in a given week he will sell 2 or more but less than 5 cars.

# In[15]:


# Probability of exactly 2 cars + exactly 3 cars + exactly 4 cars
P_n = poisson[2] + poisson[3] +poisson[4]

print("="*120)
print('Answer B. Probability that iin a given week he will sell 2 or more but less than 5 cars %1.4f' %P_n)
print("="*120)


# C. Plot the poisson distribution function for cumulative probability of cars sold per-week vs number of cars sold perweek.

# In[16]:


# Taking the cumulative of the poisson probability distribution function
cumpoisson = stats.poisson.cdf(n,rate)

#Plot the cumulativr probability distribution function
plt.plot(n,cumpoisson, 'o-')
plt.title('Poisson')
plt.ylabel('Cumulative prob. of cars sold per-week')
plt.xlabel('Number of cars sold per week')
plt.show()


# ### Question 4: Accuracy in understanding orders for a speech based bot at a restaurant is important for the Company X which has designed, marketed and launched the product for a contactless delivery due to the COVID-19 pandemic. Recognition accuracy that measures the percentage of orders that are taken correctly is 86.8%. Suppose that you place order with the bot and two friends of yours independently place orders with the same bot. 
# Answer the following questions.

# In[17]:


# Using binomial distribution to decide "success" or "failure" in recognition

p   =  0.868
n   =  3           # Me and two of my friends, so total 3 samples
k   =  np.arange(0,5)

binomial = stats.binom.pmf(k,n,p)
print(binomial)


# In[18]:


cumbinomial = stats.binom.cdf(k,n,p)
print(cumbinomial)


# A. What is the probability that all three orders will be recognised correctly?

# In[19]:


print("="*120)
print(' Answer A. Probability that all three orders will be recognised correctly %1.4f' %cumbinomial[3])
print("="*120)


# B. What is the probability that none of the three orders will be recognised correctly?

# In[20]:


print("="*120)
print(' Answer B. Probability that none of the three orders will be recognised correctly %1.4f' %binomial[0])
print("="*120)


# C. What is the probability that at least two of the three orders will be recognised correctly?

# In[21]:


P_two_or_more = binomial[2] + binomial[3]

print("="*120)
print(' Answer C. Probability that at least two of the three orders will be recognised correctly %1.4f' %P_two_or_more)
print("="*120)


# ### Question 5: A group of 300 professionals sat for a competitive exam. The results show the information of marks obtained by them have a mean of 60 and a standard deviation of 12. The pattern of marks follows a normal distribution. 
# Answer the following questions.

# In[22]:


mu = 60
sigma = 12


# A. What is the percentage of students who score more than 80.

# In[23]:


# Conversion to standard normal variable by standardization
z = (80 - mu)/sigma

# Probability of a student scoring above 80
P = 1 - stats.norm.cdf(z)

# Calculating total students scoring above 80
S = round (P*300)
S_percent = (100 * S)/300
print("="*120)
print(' Answer A. Percentage of students who score more than 80 is  %1.4f' %S_percent)
print("="*120)


# B. What is the percentage of students who score less than 50.

# In[24]:


# Conversion to standard normal variable by standardization
z = (50 - mu)/sigma

# Probability of a student scoring less than 50
P = stats.norm.cdf(z)

# Calculating total students scoring less than 50
S = round (P*300)
S_percent = (100 * S)/300
print("="*120)
print(' Answer B. Percentage of students who score less than 50 is  %1.4f' %S_percent)
print("="*120)


# C. What should be the distinction mark if the highest 10% of students are to be awarded distinction?

# * Highest 10% of students would be those students with marks above 0.9 (100% - 90% = 10%).
# * Referring the Standard Normal Distribution table 0.8997 is the closest to 0.9 and it has a corresponding z-score of 1.28 =  1.2 on yaxis and 0.08 on x axis
# 

# In[25]:


z = 1.28
x = (z*sigma)+mu

print("="*120)
print(' Answer C. The distinction mark would be  %1.4f' %x)
print("="*120)


# ### Question 6: Explain 1 real life industry scenario [other than the ones mentioned above] where you can use the concepts learnt in this module of Applied statistics to get a data driven business solution.

# #####  Domain : Electrical Engineering
# #####  Application : Componenet manufacture and Reliability Test
# ##### Problem Statement : 
# In Electrical systems real time data collection is done nowadays. A lot of data comprising of parameters, measurements, temperatures, environmental conditions, failure rates are collected.
# 
# ##### Use of Statistics and Probability Theory:
# * To get inferences from the data about variability in measurements
# * To estimate the average value of the nominal settings of the product
# * To test reliability of the product in various conditions

# ## PART 2: PROJECT BASED [Total Score : 15]

# #### DOMAIN: Sports
# #### CONTEXT: Company X manages the men's top professional basketball division of the American league system.
# The dataset contains information on all the teams that have participated in all the past tournaments. It has data
# about how many baskets each team scored, conceded, how many times they came within the first 2 positions,
# how many tournaments they have qualified, their best position in the past, etc.
# #### DATA DESCRIPTION: Basketball.csv - The data set contains information on all the teams so far participated in all the past tournaments.
# #### ATTRIBUTE INFORMATION:
# 1. Team: Team’s name
# 2. Tournament: Number of played tournaments.
# 3. Score: Team’s score so far.
# 4. PlayedGames: Games played by the team so far.
# 5. WonGames: Games won by the team so far.
# 6. DrawnGames: Games drawn by the team so far.
# 7. LostGames: Games lost by the team so far.
# 8. BasketScored: Basket scored by the team so far.
# 9. BasketGiven: Basket scored against the team so far.
# 10. TournamentChampion: How many times the team was a champion of the tournaments so far.
# 11. Runner-up: How many times the team was a runners-up of the tournaments so far.
# 12. TeamLaunch: Year the team was launched on professional basketball.
# 13. HighestPositionHeld: Highest position held by the team amongst all the tournaments played.
# 
# #### PROJECT OBJECTIVE: 
# Company’s management wants to invest on proposal on managing some of the best
# teams in the league. The analytics department has been assigned with a task of creating a report on the
# performance shown by the teams. Some of the older teams are already in contract with competitors. Hence
# Company X wants to understand which teams they can approach which will be a deal win for them.

# 1. Read the data set, clean the data and prepare a final dataset to be used for analysis.

# In[26]:


# Importing data from local
Games_data = pd.read_csv("C:/Users/INANNAR/Desktop/ABB/My Learnings/GreatLearning AIML/Project 1/DS - Part2 - Basketball.csv")
Games_data = Games_data.dropna()
Games_data.head(5)


# In[27]:


# Converting all columns symbolizing scores of some sort, into int, replacing - with zero
columns = list(Games_data)
columns = columns[1:11]
Games_data[columns] = Games_data[columns].replace({'-':0})
Games_data[columns] = Games_data[columns].astype(int)
Games_data.info()


# In[28]:


def year_filter(event):  
    # separate out the year part when launch was initiated   
    year = event[0:4]
    return year


# In[29]:


# The column of Year is modified to get just the year when launch was initiated
Games_data["TeamLaunch"] = Games_data["TeamLaunch"].apply(year_filter)
Games_data["TeamLaunch"].astype(str)
Games_data["TeamLaunch"] = pd.to_datetime(Games_data["TeamLaunch"], format="%Y")
Games_data.info()


# 2. Perform detailed statistical analysis and EDA using univariate, bi-variate and multivariate EDA techniques to get a data
# driven insights on recommending which teams they can approach which will be a deal win for them.. Also as a data
# and statistics expert you have to develop a detailed performance report using this data.

# In[30]:


# Basic Statistical Analysis
Games_data.describe()

#Comment: The statistics for 'HighestPositionHeld' can be avoided as it is a position in the tournama


# In[31]:


for column in Games_data.columns[1:9]:  
    plt.figure(figsize=(7,3))
    sns.boxplot(x=column, data=Games_data);  


# In[32]:


for column in Games_data.columns[1:9]:  
    plt.figure(figsize=(7,3))
    sns.histplot(x=column, data=Games_data);  


# In[33]:


Games_data1 = Games_data[Games_data.columns[1:9]]
sns.pairplot(Games_data1);


# In[34]:


plt.figure(figsize=(9,6))
sns.heatmap(Games_data1.corr(), annot=True);


# ### Descriptive Statistical Analysis

# In[35]:


Games_data.head()


# #### Team Performace Analysis

# In[36]:


#Calculating the percentage of wins or loses compared to the total games playes
Games_data['Winning_percent'] = Games_data['WonGames']/Games_data['PlayedGames']*100
Games_data['Draw_percent'] = Games_data['DrawnGames']/Games_data['PlayedGames']*100
Games_data['Losing_percent'] = Games_data['LostGames']/Games_data['PlayedGames']*100


# In[37]:


Winning_pecent_max = Games_data['Winning_percent'].max()
Best_Team = Games_data.loc[Games_data['Winning_percent'] == Winning_pecent_max , 'Team'].item()
print("="*120)
print("The best performing team is %s with the winning percentage of %1.2f %%" %(Best_Team, Winning_pecent_max))
print("="*120)


# In[38]:


Losing_pecent_max = Games_data['Losing_percent'].max()
Worst_Team = Games_data.loc[Games_data['Losing_percent'] == Losing_pecent_max , 'Team'].item()
print("="*120)
print("The worst performing team is %s with the losing percentage of %1.2f %%" %(Worst_Team, Losing_pecent_max))
print("="*120)


# #### Estimation of Winning percentage based on history data
# 
# Reference [https://en.wikipedia.org/wiki/Pythagorean_expectation]

# In[39]:


x = Games_data['BasketScored']**13.91
y = (Games_data['BasketScored']**13.91)+(Games_data['BasketGiven']**13.91)
Games_data['Py_Winning_percent'] = x/y*100
Games_data = Games_data.replace(np.nan, 0)


# In[40]:


Py_winning_pecent_max = Games_data['Py_Winning_percent'].max()
Best_Winning_Percent = Games_data.loc[Games_data['Py_Winning_percent'] == Py_winning_pecent_max , 'Team'].item()
print("="*120)
print("The team with the best winning percentage is %s with the winning percentage of %1.4f %%" %(Best_Winning_Percent, Py_winning_pecent_max))
print("="*120)


# ### Graphical Analysis

# In[41]:


fig, ax = plt.subplots(figsize = (12,6))    
fig = sns.barplot(x = "TeamLaunch", y = "Winning_percent", data = Games_data,color = 'blue')
fig = sns.barplot(x = "TeamLaunch", y = "Py_Winning_percent", data = Games_data,color = 'orange')

x_dates = Games_data['TeamLaunch'].dt.strftime('%Y').sort_values().unique();
ax.set_xticklabels(labels=x_dates, rotation=45, ha='right');


# #### Inference: The winning percentage is not very largely affected by the age/experience of the team 

# In[42]:


fig, ax = plt.subplots(figsize = (12,6))    
fig = sns.barplot(x = "TeamLaunch", y = "TournamentChampion", data = Games_data,color = 'blue')
fig = sns.barplot(x = "TeamLaunch", y = "Runner-up", data = Games_data,color = 'orange')

x_dates = Games_data['TeamLaunch'].dt.strftime('%Y').sort_values().unique();
ax.set_xticklabels(labels=x_dates, rotation=45, ha='right');


# #### Inference : The tournament champions or runners up are the teams that were established many years ago and have good experience in playing, also probably because they have played more number of games hence ,more chances of winning/losing

# 3. Please include any improvements or suggestions to the association management on quality, quantity, variety, velocity,
# veracity etc. on the data points collected by the association to perform a better data analysis in future.

# #### Volume : Suggested improvements in the data volume
# * Information about all matches conducted since 90s would be a great source of information for detailed analysis.
# 
# #### Velocity : Suggested improvements in the data velocity/frequency of collection
# * Information about the various matches held across the years could add more value to the data analysis.
# 
# #### Variety : Suggested improvements in the data variety/type of data sources available
# * Place of matches , Player informations, Tournaments and Games details would add more value
# 
# #### Veracity : Suggested improvements in the data quality
# * Dates of team launch could be the exact date of team registration
# 
# #### Value : Suggested improvements in the value addition
# * A large scope of data collection for detailed game, team performace and player performance is there

# ## PART 3: PROJECT BASED [Total Score : 30]

# #### DOMAIN: Startup ecosystem
# #### CONTEXT: 
# Company X is a EU online publisher focusing on the startups industry. The company specifically reports on the business related to technology news, analysis of emerging trends and profiling of new tech businesses and products. Their event i.e. Startup Battlefield is the world’s pre-eminent startup competition. Startup Battlefield features 15-30 top early stage startups pitching top judges in front of a vast live audience, present in person and online.
# #### DATA DESCRIPTION: CompanyX_EU.csv - Each row in the dataset is a Start-up company and the columns describe the company.
# #### ATTRIBUTE INFORMATION:
# 1. Startup: Name of the company
# 2. Product: Actual product
# 3. Funding: Funds raised by the company in USD
# 4. Event: The event the company participated in
# 5. Result: Described by Contestant, Finalist, Audience choice, Winner or Runner up
# 6. OperatingState: Current status of the company, Operating ,Closed, Acquired or IPO
# *Dataset has been downloaded from the internet. All the credit for the dataset goes to the original creator of the data.
# 
# #### PROJECT OBJECTIVE: 
# Analyse the data of the various companies from the given dataset and perform the tasks that are specified in the below steps. Draw insights from the various attributes that are present in the dataset, plot distributions, state hypotheses and draw conclusions from the dataset.

# ##### 1. Data warehouse:
# * Read the CSV file.

# In[43]:


# Importing data from local
CompanyX = pd.read_csv("C:/Users/INANNAR/Desktop/ABB/My Learnings/GreatLearning AIML/Project 1/DS - Part3 - CompanyX_EU.csv")
CompanyX.head(5)


# ##### 2. Data exploration:
# * Check the datatypes of each attribute.
# * Check for null values in the attributes

# In[44]:


CompanyX.info()


# ##### 3. Data preprocessing & visualisation:
# * Drop the null values.
# * Convert the ‘Funding’ features to a numerical value.
# * Plot box plot for funds in million.
# * Get the lower fence from the box plot.
# * Check number of outliers greater than upper fence.
# * Drop the values that are greater than upper fence.
# * Plot the box plot after dropping the values.
# * Check frequency of the OperatingState features classes.
# * Plot a distribution plot for Funds in million.
# * Plot distribution plots for companies still operating and companies that closed.

# ##### 3a. Drop the null values.

# In[45]:


#Dropping the null values
CompanyX = CompanyX.dropna()
CompanyX.info()


# ##### 3b. Convert the ‘Funding’ features to a numerical value.

# In[46]:


CompanyX_EU = CompanyX.copy()
#Converting the column entries to string
CompanyX_EU["Funding"] = CompanyX["Funding"].astype(str)

#Removing dolar symbol '$' from the column entries
CompanyX_EU["Funding"] = CompanyX_EU["Funding"].str.replace('$','')


# In[47]:


# This is a user-defined function for converting the present entries of 
def conversion(currency):
    
    #Reference dictionary for key value pair defining symbol to multiplier value
    d = {'K': 1000,'M': 1000000,'B': 1000000000}
    
    if currency[-1] in d:
        # separate out the K, M, or B
        number, multiplier = currency[:-1], currency[-1]
        return int(float(number) * d[multiplier])
    else:
        return float(currency)


# In[48]:


CompanyX_EU["Funding"] = CompanyX_EU["Funding"].apply(conversion)


# In[49]:


#Intermediate view of updated columns
CompanyX_EU["Funding"].dtype


# ##### 3c. Plot box plot for funds in million.

# In[50]:


# Converting 'Funding' column to millions 
CompanyX_EU["Funding_In_Millions"] = CompanyX_EU["Funding"]/1e6 
CompanyX_EU


# In[51]:


# Box plot for 'Funding in Millions'
fig, ax = plt.subplots(figsize=(10,5))
sns.boxplot(x='Funding_In_Millions',data=CompanyX_EU).set_title("Distribution of Funding in Million Dolars");


# ##### 3d.Get the lower fence from the box plot.

# In[52]:


#Calculating 25th percentile
Q1 = np.percentile(CompanyX_EU["Funding_In_Millions"], 25)
Q3 = np.percentile(CompanyX_EU["Funding_In_Millions"], 75)

#Calculating the InterQuatileRange
IQR = Q3-Q1

print("The 25th percentile value = %1.4f" %Q1)
print("The 75th percentile value = %1.4f" %Q3)
print("The Inter Quatile Range value = %1.4f" %IQR)


# In[53]:


#The lower fence of the box plot is given by
Lower_fence = Q1 - (1.5 * IQR)
print("The value of the lower fence of the box plot = %1.4f" %Lower_fence)


# ##### 3e. Check number of outliers greater than upper fence.

# In[54]:


#The upper fence of the box plot is given by
Upper_fence = Q3 + (1.5 * IQR)
print("The value of the upper fence of the box plot = %1.4f" %Upper_fence)


# ##### 3f. Drop the values that are greater than upper fence.

# In[55]:


#Extracting values satisfying the condition of CompanyX_EU["Funding_In_Millions"] is less than or equal to Upper_fence
Funding_Mil = CompanyX_EU[CompanyX_EU["Funding_In_Millions"] <= Upper_fence]
Funding_Mil


# ##### 3g. Plot the box plot after dropping the values.

# In[56]:


# Box plot for 'Funding in Millions'
fig, ax = plt.subplots(figsize=(10,5))
sns.boxplot(x='Funding_In_Millions',data=Funding_Mil).set_title("Distribution of Funding in Million Dolars");


# ##### 3h. Check frequency of the OperatingState features classes.

# In[57]:


#Plotting the categorical data and the count of each
fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(y="OperatingState", data=CompanyX_EU)


# ##### 3i. Plot a distribution plot for Funds in million.

# In[58]:


#Plotting the distribution of the data using histplot
sns.histplot(CompanyX_EU["Funding_In_Millions"]).set_title("Distribution of Funding(Mil) with Outliers");
plt.show()

sns.histplot(Funding_Mil["Funding_In_Millions"]).set_title("Distribution of Funding(Mil) without Outliers");
plt.show()


# ##### 3j. Plot distribution plots for companies still operating and companies that closed.

# In[59]:


#Filtering out the companies still operating including outliers
Companies_operating_o = CompanyX_EU[CompanyX_EU["OperatingState"] == 'Operating']
Companies_operating_o.head(5)

#Plotting the distribution of the data using histplot
sns.histplot(Companies_operating_o["Funding_In_Millions"]).set_title("Distribution of Funding(Mil) with Outliers");
plt.show()


# In[60]:


#Filtering out the companies still operating excluding outliers
Companies_operating = Funding_Mil[Funding_Mil["OperatingState"] == 'Operating']
Companies_operating.head(5)

#Plotting the distribution of the data using histplot
sns.histplot(Companies_operating["Funding_In_Millions"]).set_title("Distribution of Funding(Mil) without Outliers");
plt.show()


# In[61]:


#Filtering out the companies that are closed including outliers
Companies_closed_o = CompanyX_EU[CompanyX_EU["OperatingState"] == 'Closed']
Companies_closed_o.head(5)

#Plotting the distribution of the data using histplot
sns.histplot(Companies_closed_o["Funding_In_Millions"]).set_title("Distribution of Funding(Mil) with Outliers");
plt.show()


# In[62]:


#Filtering out the companies that are closed excluding outliers
Companies_closed = Funding_Mil[Funding_Mil["OperatingState"] == 'Closed']
Companies_closed.head(5)

#Plotting the distribution of the data using histplot
sns.histplot(Companies_closed["Funding_In_Millions"]).set_title("Distribution of Funding(Mil) without Outliers");
plt.show()


# ##### 4. Statistical analysis:
# * Is there any significant difference between Funds raised by companies that are still operating vs companies that closed down?
#     * Write the null hypothesis and alternative hypothesis.
#     * Test for significance and conclusion
# * Make a copy of the original data frame.
# * Check frequency distribution of Result variable.
# * Calculate percentage of winners that are still operating and percentage of contestants that are still operating
# * Write your hypothesis comparing the proportion of companies that are operating between winners and contestants:
#     * Write the null hypothesis and alternative hypothesis.
#     * Test for significance and conclusion
# * Check distribution of the Event variable.
# * Select only the Event that has disrupt keyword from 2013 onwards.
# * Write and perform your hypothesis along with significance test comparing the funds raised by companies across NY, SF and EU events from 2013 onwards.
# * Plot the distribution plot comparing the 3 city events.

# ##### 4a. Is there any significant difference between Funds raised by companies that are still operating vs companies that closed down?
# * Write the null hypothesis and alternative hypothesis.
# * Test for significance and conclusion

# In[63]:


#Case 1 is considering the outliers

#Sample 1 is funding_in_millions column for companies that are operating
Sample_01 = Companies_operating_o["Funding_In_Millions"]

#Sample 2 is funding_in_millions column for companies that are closed
Sample_02 = Companies_closed_o["Funding_In_Millions"]


# Comments:
# * In this case T - test for 2 samples will be applicable.
# * The two samples are independent - Closed vs Operating companies
# * The H0 = NULL HYPOTHESIS is "The two samples have the same mean"
# * The Ha = ALTERNATE HYPOTHESIS is "The two samples do not have same mean at 5% significance level"
# * Alternatively the ALTERNATE HYPOTHESIS is p_value < 0.05 

# In[64]:


#Computing the T test and finding the p-value
t_statistic, pvalue = ttest_ind(Sample_01, Sample_02)
print("The two sample T-test p value (considering outliers in funding value) = ", pvalue)


# In[65]:


#Case 2 is without considering the outliers

#Sample 1 is funding_in_millions column for companies that are operating
Sample_11 = Companies_operating["Funding_In_Millions"]

#Sample 2 is funding_in_millions column for companies that are closed
Sample_21 = Companies_closed["Funding_In_Millions"]


# In[66]:


#Computing the T test and finding the p-value
t_statistic, pvalue = ttest_ind(Sample_11, Sample_21)
print("The two sample T-test p value (not considering outliers in funding value) = ", pvalue)


# ### Concluding Statements:
# " Since in both the cases (with and without outliers) the p-value is greater than type 1 error rate i.e. 0.05 we fail to reject the null hypothesis "
# 
# ### Hence we can say that there is not much difference in the mean funding amounts of the companies that are operating versus the ones that are closed

# ##### 4b. Make a copy of the original data frame.

# In[67]:


Startup_Company = CompanyX.copy()
Startup_Company.head(5)


# ##### 4c. Check frequency distribution of Result variable.

# In[68]:


#Plotting the categorical data and the count of each
fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(y="Result", data=Startup_Company)


# ##### 4d. Calculate percentage of winners that are still operating and percentage of contestants that are still operating

# In[69]:


#Filtering out the companies that are Winners
St_filtered = Startup_Company[Startup_Company['Result'] == 'Winner']

#Total count of each OperatingState category
Count_data = St_filtered['OperatingState'].value_counts()
print(Count_data)


# In[70]:


# Percentage of Winner still operating
Operating_Winner = (Count_data[0]/Count_data.sum())*100
print("Percentage of winning companies that are still operating = ", Operating_Winner)


# In[71]:


#Filtering out the companies that are Contestants
St_filtered = Startup_Company[Startup_Company['Result'] == 'Contestant']

#Total count of each Result category
Count_data = St_filtered['OperatingState'].value_counts()
print(Count_data)


# In[72]:


# Percentage of Contestant still operating
Operating_Contestant = (Count_data[0]/Count_data.sum())*100
print("Percentage of companies that contested and are still operating = ", Operating_Contestant)


# ##### 4e. Write your hypothesis comparing the proportion of companies that are operating between winners and contestants:
# * Write the null hypothesis and alternative hypothesis.
# * Test for significance and conclusion

# In[73]:


winner_operating = Startup_Company[Startup_Company['Result'] == 'Winner'].OperatingState.value_counts()[0]  # number of operating winners
contestant_operating = Startup_Company[Startup_Company['Result'] == 'Contestant'].OperatingState.value_counts()[0] # number of operating contestants
n_winner = Startup_Company.Result.value_counts()[3] # number of winners in the data
n_contestant = Startup_Company.Result.value_counts()[0] #number of contestants in the data


# Comments:
# * The Null Hypothesis Ho = "The proportions of operating companies between winner vs contestants are equal"
# * The Alternate Hypothesis Ha = "The proportions of operating companies are not equal"

# In[74]:


print([winner_operating, contestant_operating] , [n_winner, n_contestant])
print(f' Proportion of operating companies in Winners,Contestants = {round(winner_operating/n_winner,2)}, {round(contestant_operating/n_contestant,2)} respectively')


# In[75]:


statistics, pval = proportions_ztest([winner_operating, contestant_operating] , [n_winner, n_contestant])
print("The z-test p value = ", pvalue)


# ### Concluding Statements:
# " Since the p-value is greater than type 1 error rate i.e. 0.05 we fail to reject the null hypothesis "
# 
# ### Hence we can say that there is no  significant difference in the two proportions - proportion of companies between winner vs contestants

# ##### 4f. Check distribution of the Event variable.

# In[76]:


Events = CompanyX.copy()
#Plotting the categorical data and the count of each
fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(y="Event", data=Events)


# In[77]:


# As observed from the above graph, few of the events have no name, so dropping those events from the analysis
Events = Events[Events['Event'] != '-']
Events

#Plotting the categorical data and the count of each
fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(y="Event", data=Events)


# ##### 4g. Select only the Event that has disrupt keyword from 2013 onwards.

# In[78]:


# This is a user-defined function for converting the present entries of 
def year_filter(event):  
    # separate out the year part    
    year = event[-4:]
    return int(year)    


# In[79]:


# Another column of Year is created from the event column which shall be used as filter
Events["Year"] = Events["Event"].apply(year_filter)
Events.head()


# In[80]:


#Filtering out the events that contains disrupt and occured from 2013 onwards
Events = Events[(Events['Event'].str.contains("Disrupt")) & (Events['Year'] >= 2013)]
Events.head()


# In[81]:


#Verifiying with the unique Events and Year values
Events['Event'].unique()
Events['Year'].unique()


# ##### 4h. Write and perform your hypothesis along with significance test comparing the funds raised by companies across NY, SF and EU events from 2013 onwards.

# In[82]:


#Filtering out the events that contains 'NY' 'SF' or 'EU' and occured from 2013 onwards
Events = Events[(Events['Event'].str.contains('NY | SF | EU')) & (Events['Year'] >= 2013)]
Events['Event'].unique()


# In[83]:


#Some preprocessing steps before analysis

#Converting the column entries to string
Events["Funding"] = Events["Funding"].astype(str)

#Removing dolar symbol '$' from the column entries
Events["Funding"] = Events["Funding"].str.replace('$','')

#Getting the Funding_in_millions column
Events["Funding"] = Events["Funding"].apply(conversion)
Events["Funding_in_millions"] = Events["Funding"]/1e6


# In[84]:


# Separating the three samples - Funding for NY, SF and EU
Event_in_NY = Events[Events['Event'].str.contains('NY')]['Funding_in_millions']
Event_in_SF = Events[Events['Event'].str.contains('SF')]['Funding_in_millions']
Event_in_EU = Events[Events['Event'].str.contains('EU')]['Funding_in_millions']


# In[85]:


# Creating a common dataframe
Funding_df = pd.DataFrame()

df1 = pd.DataFrame({'Event_In': 'NY', 'Funding_Mil':Event_in_NY})
df2 = pd.DataFrame({'Event_In': 'SF', 'Funding_Mil':Event_in_SF})
df3 = pd.DataFrame({'Event_In': 'EU', 'Funding_Mil':Event_in_EU})

Funding_df = Funding_df.append(df1) 
Funding_df = Funding_df.append(df2) 
Funding_df = Funding_df.append(df3)
Funding_df.head()


# In[86]:


#Comparing the data spread for each case
fig, ax = plt.subplots(figsize=(10,5))
sns.boxplot(x = "Event_In", y = "Funding_Mil", data = Funding_df)
plt.title('Funding received for the events in NY, SF and EU')
plt.show()


# #### Comments:
# * The null hypothesis Ho is mu1 = mu2 = mu3 i.e. the mean funding for the events in NY , SF and EU are same
# * The alternate hypothesis Ha is that atleast one mu differs
# * The significance level is alpha = 0.05

# In[87]:


#Creating a model from a formula and dataframe
#Creating Anova table for one or more fitted linear models

mod = ols('Funding_Mil ~ Event_In', data = Funding_df).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
### Concluding Statements:
" Since in both the cases (with and without outliers) the p-value is greater than type 1 error rate i.e. 0.05 we fail to reject the null hypothesis "

### Hence we can say that there is not much difference in the mean funding amounts of the companies that are operating versus the ones that are closedprint(aov_table)


# ### Concluding Statements:
# " In this example, p value is 0.40646 and it is greater than our chosen level of signifance at 5% 
# 
# So the statistical decision is that we fail to reject the null hypothesis at 5% level of significance."
# 
# ### Hence we can say that there is not much difference in the mean funding amounts of the companies for the events that took place in NY, SF and EU
# 
# 
# 

# ##### 4i. Plot the distribution plot comparing the 3 city events.

# In[88]:


#Plotting the distribution of the data using histplot
fig, ax = plt.subplots(figsize=(15,10))
sns.histplot(x = "Funding_Mil",hue = 'Event_In', data = Funding_df).set_title("Distribution of Funding(Mil)");
plt.show()


# ##### 5. Write your observations on improvements or suggestions on quality, quantity, variety, velocity, veracity etc. on the data points collected to perform a better data analysis.

# #### Volume : Suggested improvements in the data volume
# * Since the company's target was to analysis new technologies, profiling new tech businesses, it would be nice to target a larger audience. The event column suggested very limited places where the competitions were held. That can be expanded to cover comapnies from other provinces
# 
# #### Velocity : Suggested improvements in the data velocity/frequency of collection
# * The frequency of conducting the events can be increased to get more data and also more participation.
# 
# #### Variety : Suggested improvements in the data variety/type of data sources available
# * An additional column suggesting product categorization could be added, suggesting the domain that the product covers like - Electrical or Finance or Food Industry. That would suggest many things about the growing industry and the industries that are popular for startups
# 
# #### Veracity : Suggested improvements in the data quality
# * Around 30% of the row entries has missing entries for the various attributes hence a good amount of data was not considered in the analysis. This number can be reduced by online data collection, mandate entries for all columns etc.
# 
# #### Value : Suggested improvements in the value addition
# * More information about the product like price or target industry could be added. The existing data gives very less information about the emerging technology which is one of the prime focus of the company
# 
# 
# 

# In[ ]:




